"""Settings files: reading them, stacking them, and writing the result (ADR-0008).

Settings are built in layers. Each layer changes only the settings it mentions:

1. the defaults in `mlracecar.config.models`,
2. settings files, in the order given,
3. ``--set section.key=value`` overrides from the command line, in the order given.

Files are YAML, read safely (no code runs) with two fixes over plain YAML 1.1: numbers like
``1e3`` read as numbers rather than text, and a setting written twice in one file is an error
instead of silently keeping the second.

Example::

    # heavy-car.yaml
    vehicle:
      mass: 1800
"""

import difflib
import json
import re
from collections.abc import Hashable, Iterable, Mapping
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, ValidationError
from pydantic.fields import FieldInfo
from pydantic_core import ErrorDetails

from mlracecar.config.models import RacecarConfig

HEADER = (
    "# MLRacecar settings: every setting with its value.\n"
    "# To change a few, write a file with just those lines (under their section), or use\n"
    "# --set, e.g. `racecar config --set vehicle.mass=1500`.\n"
)
"""The comment at the top of every settings file MLRacecar writes."""

type Location = tuple[Hashable, ...]
"""Where a value sits in the settings, e.g. ``("vehicle", "mass")``."""


class ConfigError(ValueError):
    """Settings can't be used. The message lists every problem and where it came from."""


def load_config(files: Iterable[str | Path] = (), overrides: Iterable[str] = ()) -> RacecarConfig:
    """Build the settings: the defaults, changed by each file in order, then by each override.

    Args:
        files: Settings files (YAML); each only needs the settings it changes.
        overrides: ``section.key=value`` strings, as given to ``--set``.

    Raises:
        ConfigError: If a file can't be read, an override isn't ``key=value``, or a setting is
            invalid. Every invalid setting is listed, with the file (or ``--set``) it came from.
    """
    layers = [(str(path), read_config_file(path)) for path in files]
    layers += [("--set", parse_override(override)) for override in overrides]
    merged: dict[Hashable, Any] = {}
    sources: dict[Location, str] = {}
    for source, data in layers:
        merged = _merge(merged, data, source, sources)
    try:
        return RacecarConfig.model_validate(merged)
    except ValidationError as error:
        raise ConfigError(_describe(error, sources)) from None


def read_config_file(path: str | Path) -> dict[Hashable, Any]:
    """Read one settings file as nested dictionaries, without checking the settings yet.

    Raises:
        ConfigError: If the file can't be read or isn't YAML with sections at the top level.
    """
    path = Path(path)
    try:
        # As bytes, so YAML can tell UTF-8 from UTF-16 (what Windows PowerShell 5.1's `>` writes).
        content = path.read_bytes()
    except OSError as error:
        raise ConfigError(f"{path}: can't read the file ({error.strerror})") from error
    return parse_config_text(content, source=str(path))


def parse_config_text(text: str | bytes, source: str = "settings") -> dict[Hashable, Any]:
    """Read the text of a settings file; ``source`` names it in error messages.

    Bytes may be UTF-8, or UTF-16 with a byte-order mark.

    Raises:
        ConfigError: If the text isn't YAML with sections at the top level.
    """
    data = _parse_yaml(text, source)
    if data is None:  # an empty file changes nothing
        return {}
    if not isinstance(data, dict):
        raise ConfigError(
            f"{source}: expected sections of settings (like `vehicle:`), got {_show(data)}"
        )
    return data


def parse_override(text: str) -> dict[Hashable, Any]:
    """Turn ``section.key=value`` into nested dictionaries. The value is read as YAML.

    ``vehicle.mass=1500`` becomes ``{"vehicle": {"mass": 1500}}``.

    Raises:
        ConfigError: If the text isn't ``key=value`` or the value isn't valid YAML.
    """
    key, equals, value = text.partition("=")
    parts = [part.strip() for part in key.split(".")]
    if not equals or not all(parts):
        raise ConfigError(f"--set {text}: expected section.key=value, e.g. vehicle.mass=1500")
    data: dict[Hashable, Any] = {parts[-1]: _parse_yaml(value, source=f"--set {text}")}
    for part in reversed(parts[:-1]):
        data = {part: data}
    return data


def format_config(config: RacecarConfig) -> str:
    """The settings as YAML, with a short explanation on every line.

    Reading the text back (`parse_config_text`) gives exactly the same settings.
    """
    return HEADER + "\n".join(_section_lines(config, indent="")) + "\n"


# --------------------------------------------------------------------------- #
# YAML
# --------------------------------------------------------------------------- #


class _SettingsLoader(yaml.SafeLoader):
    """PyYAML's safe loader, with numbers like ``1e3`` and repeated keys fixed (see above)."""

    def construct_mapping(self, node: yaml.MappingNode, deep: bool = False) -> dict[Hashable, Any]:
        seen: set[Hashable] = set()
        for key_node, _ in node.value:
            if key_node.tag == "tag:yaml.org,2002:merge":
                continue  # `<<: *defaults` merge keys may be overridden by design
            key = self.construct_object(key_node, deep=deep)
            if isinstance(key, Hashable):
                if key in seen:
                    raise yaml.constructor.ConstructorError(
                        None, None, f"{key} is written twice", key_node.start_mark
                    )
                seen.add(key)
        return super().construct_mapping(node, deep)


# YAML 1.1 only reads a number with an exponent as a number if it has a dot and a signed
# exponent (1.0e+3); YAML 1.2, and people, also write 1e3 and 2.5E-4.
_SettingsLoader.add_implicit_resolver(
    "tag:yaml.org,2002:float",
    re.compile(r"^[-+]?(?:[0-9][0-9_]*(?:\.[0-9_]*)?|\.[0-9_]+)[eE][-+]?[0-9]+$"),
    list("-+0123456789."),
)


def _parse_yaml(text: str | bytes, source: str) -> Any:
    try:
        return yaml.load(text, Loader=_SettingsLoader)  # a SafeLoader: no code runs
    except yaml.reader.ReaderError as error:
        raise ConfigError(
            f"{source}: not readable text ({error.reason} at position {error.position})"
        ) from None
    except yaml.MarkedYAMLError as error:
        mark = error.problem_mark
        where = f" (line {mark.line + 1}, column {mark.column + 1})" if mark else ""
        raise ConfigError(f"{source}: not valid YAML{where}: {error.problem}") from None


# --------------------------------------------------------------------------- #
# Stacking layers
# --------------------------------------------------------------------------- #


def _merge(
    base: Mapping[Hashable, Any],
    update: Mapping[Hashable, Any],
    source: str,
    sources: dict[Location, str],
    prefix: Location = (),
) -> dict[Hashable, Any]:
    """``base`` with ``update`` laid on top: sections merge key by key; anything else replaces.

    Records in ``sources`` which layer set each value, for error messages. Only values are
    recorded, not sections, so a default inside a section someone changed has no source.
    """
    merged = dict(base)
    for key, value in update.items():
        location = (*prefix, key)
        earlier = merged.get(key)
        if isinstance(value, dict):
            if not isinstance(earlier, dict):
                _forget(sources, location)
                earlier = {}
            merged[key] = _merge(earlier, value, source, sources, location)
        else:
            _forget(sources, location)
            merged[key] = value
            sources[location] = source
    return merged


def _forget(sources: dict[Location, str], location: Location) -> None:
    """Drop the sources of a value, or of every value in a section, that is being replaced."""
    for known in _within(sources, location):
        del sources[known]


def _within(sources: Mapping[Location, str], location: Location) -> list[Location]:
    """The recorded locations at ``location`` or inside it, oldest first."""
    return [known for known in sources if known[: len(location)] == location]


# --------------------------------------------------------------------------- #
# Error messages
# --------------------------------------------------------------------------- #


def _describe(error: ValidationError, sources: Mapping[Location, str]) -> str:
    lines = ["Invalid settings:"]
    for issue in error.errors():
        location = issue["loc"]
        line = f"  {'.'.join(str(part) for part in location)}: {_problem(issue)}"
        source = _source_of(location, sources)
        lines.append(f"{line} (from {source})" if source else line)
    return "\n".join(lines)


def _problem(issue: ErrorDetails) -> str:
    """Pydantic's error, in plain words."""
    limits = issue.get("ctx") or {}
    got = f", got {_show(issue['input'])}"
    match issue["type"]:
        case "extra_forbidden":
            return "unknown setting" + _suggestion(issue["loc"])
        case "invalid_key":
            return "setting names must be text"
        case "model_type":
            return f"must be a section of settings (key: value lines){got}"
        case "float_type":
            return f"must be a number{got}"
        case "int_type":
            return f"must be a whole number{got}"
        case "finite_number":
            return f"must be a finite number{got}"
        case "greater_than":
            return f"must be greater than {_format_scalar(limits['gt'])}{got}"
        case "greater_than_equal":
            return f"must be at least {_format_scalar(limits['ge'])}{got}"
        case "less_than":
            return f"must be less than {_format_scalar(limits['lt'])}{got}"
        case "literal_error":
            return f"must be one of {limits['expected']}{got}"
        case "bool_type":
            return f"must be true or false{got}"
        case "no_inputs":
            return issue["msg"]  # the whole section was the input: no use repeating it
        case _:
            return issue["msg"] + got


def _suggestion(location: tuple[int | str, ...]) -> str:
    """``"; did you mean …?"`` for a misspelled setting, or nothing if no name is close."""
    model: type[BaseModel] = RacecarConfig
    for part in location[:-1]:
        section = model.model_fields[str(part)].annotation
        assert isinstance(section, type)
        assert issubclass(section, BaseModel)
        model = section
    close = difflib.get_close_matches(str(location[-1]), model.model_fields, n=1)
    return f"; did you mean {close[0]}?" if close else ""


def _source_of(location: Location, sources: Mapping[Location, str]) -> str | None:
    """The layer that set the value at ``location`` (for a section, the layers that set any of
    its values), or ``None`` for a default."""
    layers = dict.fromkeys(sources[known] for known in _within(sources, location))
    return ", ".join(layers) or None


def _show(value: object) -> str:
    """A value as the user would have written it."""
    if value is None:
        return "nothing"
    try:
        return json.dumps(value, default=str, ensure_ascii=False)
    except (TypeError, ValueError):  # keys JSON can't show, or a YAML structure containing itself
        return str(value)


# --------------------------------------------------------------------------- #
# Writing
# --------------------------------------------------------------------------- #


def _section_lines(section: BaseModel, indent: str) -> list[str]:
    fields = type(section).model_fields
    values = {name: getattr(section, name) for name in fields}
    rows = {
        name: f"{name}: {_format_scalar(value)}"
        for name, value in values.items()
        if not isinstance(value, BaseModel)
    }
    width = max((len(row) for row in rows.values()), default=0) + 2
    lines: list[str] = []
    for name, value in values.items():
        comment = _comment(fields[name])
        if name in rows:
            lines.append(f"{indent}{rows[name]:<{width}}{comment}".rstrip())
        else:
            lines += ["", f"{indent}{comment}", f"{indent}{name}:"]
            lines += _section_lines(value, indent + "  ")
    return lines


def _comment(field: FieldInfo) -> str:
    return f"# {' '.join(field.description.split())}" if field.description else ""


def _format_scalar(value: object) -> str:
    """A setting's value as YAML. Floats keep every digit, so they read back exactly."""
    if isinstance(value, bool):  # before int: True is an int too
        return "true" if value else "false"
    if isinstance(value, float):
        return repr(value).removesuffix(".0")  # 1300.0 reads back as 1300.0 either way
    if isinstance(value, int):
        return str(value)
    if isinstance(value, str):
        # Plain if YAML reads it back as the same text; quoted otherwise ("on" is true).
        return value if _parse_yaml(value, "") == value else json.dumps(value)
    raise TypeError(f"no YAML form for {type(value).__name__} settings yet")

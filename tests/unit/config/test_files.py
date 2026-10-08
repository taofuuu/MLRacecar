"""Tests for mlracecar.config.files: stacking layers, clear errors, and lossless round trips."""

from pathlib import Path
from typing import Any

import pytest
from hypothesis import given
from hypothesis import strategies as st
from pydantic import BaseModel

from mlracecar.config.files import (
    ConfigError,
    format_config,
    load_config,
    parse_config_text,
    parse_override,
    read_config_file,
)
from mlracecar.config.models import (
    OBSERVATION_INPUTS,
    REWARD_TERMS,
    EpisodeConfig,
    ObservationConfig,
    PPOConfig,
    RacecarConfig,
    RaceConfig,
    RewardConfig,
    SensorConfig,
    SimulationConfig,
    TrainingConfig,
    VehicleConfig,
)

DEFAULT_CONFIG_PATH = Path(__file__).parents[3] / "configs" / "default.yaml"


def write(folder: Path, name: str, text: str) -> Path:
    path = folder / name
    path.write_text(text, encoding="utf-8")
    return path


def problems(files: list[Path] | None = None, overrides: list[str] | None = None) -> list[str]:
    """The lines of the error that loading these layers raises."""
    with pytest.raises(ConfigError) as caught:
        load_config(files or [], overrides or [])
    header, *lines = str(caught.value).splitlines()
    assert header == "Invalid settings:"
    return [line.strip() for line in lines]


# --------------------------------------------------------------------------- #
# Layers: defaults < files < --set
# --------------------------------------------------------------------------- #


def test_no_layers_give_the_defaults() -> None:
    assert load_config() == RacecarConfig()


def test_default_file_is_up_to_date() -> None:
    assert DEFAULT_CONFIG_PATH.read_text(encoding="utf-8") == format_config(RacecarConfig()), (
        "configs/default.yaml is out of date: run `uv run python scripts/export_default_config.py`"
    )


def test_default_file_holds_the_defaults() -> None:
    assert load_config([DEFAULT_CONFIG_PATH]) == RacecarConfig()


def test_layers_apply_in_order_each_changing_only_what_it_mentions(tmp_path: Path) -> None:
    first = write(tmp_path, "first.yaml", "vehicle:\n  mass: 1500\n  width: 1.8\n")
    second = write(tmp_path, "second.yaml", "vehicle:\n  mass: 1600\n  length: 4.2\n")

    config = load_config([first, second], ["vehicle.length=4.0", "simulation.physics_hz=240"])

    assert config.vehicle.width == 1.8  # from the first file
    assert config.vehicle.mass == 1600.0  # the second file wins over the first
    assert config.vehicle.length == 4.0  # --set wins over the files
    assert config.simulation.physics_hz == 240
    assert config.vehicle.wheelbase == VehicleConfig().wheelbase  # nobody changed it
    assert config.simulation.action_repeat == SimulationConfig().action_repeat


def test_settings_can_start_from_others_instead_of_the_defaults() -> None:
    saved = RacecarConfig.model_validate({"vehicle": {"mass": 1500}, "episode": {"time_limit": 30}})
    base = ("the model card", saved.model_dump(mode="json"))

    assert load_config(base=base) == saved
    config = load_config(overrides=["episode.time_limit=90"], base=base)
    assert (config.vehicle.mass, config.episode.time_limit) == (1500.0, 90.0)
    partial = load_config(base=("old settings", {"vehicle": {"mass": 1400}}))
    assert partial.vehicle.width == VehicleConfig().width  # left out: the default


def test_a_problem_in_the_settings_started_from_says_where_they_came_from() -> None:
    with pytest.raises(ConfigError) as caught:
        load_config(base=("the model card", {"vehicle": {"mass": -1, "colour": "red"}}))

    assert str(caught.value).splitlines() == [
        "Invalid settings:",
        "  vehicle.mass: must be greater than 0, got -1 (from the model card)",
        "  vehicle.colour: unknown setting (from the model card)",
    ]


def test_the_last_set_of_a_setting_wins() -> None:
    config = load_config(overrides=["vehicle.mass=1500", "vehicle.mass=1700"])

    assert config.vehicle.mass == 1700.0


@pytest.mark.parametrize("text", ["", "# nothing to change\n", "{}\n"])
def test_an_empty_file_changes_nothing(tmp_path: Path, text: str) -> None:
    assert load_config([write(tmp_path, "empty.yaml", text)]) == RacecarConfig()


# --------------------------------------------------------------------------- #
# --set
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    ("text", "value"),
    [
        ("vehicle.mass=1500", 1500),
        ("vehicle.mass=1.5e3", 1500.0),
        ("vehicle.mass=-2", -2),
        ("vehicle.mass = 7", 7),
        ("vehicle.mass=", None),
        ("vehicle.mass=heavy", "heavy"),
    ],
)
def test_set_values_are_read_as_yaml(text: str, value: Any) -> None:
    assert parse_override(text) == {"vehicle": {"mass": value}}


def test_set_can_replace_a_whole_section() -> None:
    assert parse_override("vehicle={mass: 1500}") == {"vehicle": {"mass": 1500}}


@pytest.mark.parametrize("text", ["vehicle.mass", "=5", "vehicle..mass=5", "vehicle.=5", ""])
def test_set_needs_a_key_and_a_value(text: str) -> None:
    with pytest.raises(ConfigError, match=r"expected section\.key=value, e\.g\. vehicle\.mass="):
        parse_override(text)


def test_set_reports_unreadable_values() -> None:
    with pytest.raises(ConfigError, match=r"--set vehicle\.mass=\[1: not valid YAML"):
        parse_override("vehicle.mass=[1")


# --------------------------------------------------------------------------- #
# Reading YAML
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    ("text", "value"),
    [
        ("1e3", 1000.0),
        ("2.5E-4", 0.00025),
        ("-1e+2", -100.0),
        (".5e1", 5.0),
        ("1.0e+3", 1000.0),
        ("1.5", 1.5),
        ("1_000", 1000),
        ("7", 7),
        ("1e3x", "1e3x"),
    ],
)
def test_numbers_read_as_people_write_them(text: str, value: object) -> None:
    assert parse_config_text(f"vehicle:\n  mass: {text}\n") == {"vehicle": {"mass": value}}


def test_a_setting_written_twice_is_an_error() -> None:
    with pytest.raises(ConfigError, match=r"car\.yaml: not valid YAML \(line 3, column 3\): mass"):
        parse_config_text("vehicle:\n  mass: 1500\n  mass: 1600\n", source="car.yaml")


def test_yaml_merge_keys_still_work() -> None:
    text = "base: &base\n  mass: 1500\n  width: 1.8\nvehicle:\n  <<: *base\n  mass: 1600\n"

    assert parse_config_text(text)["vehicle"] == {"mass": 1600, "width": 1.8}


def test_a_list_cannot_name_a_setting() -> None:
    with pytest.raises(ConfigError, match=r"not valid YAML \(line 1, column 1\): found unhashable"):
        parse_config_text("[1]: 2\n")


def test_broken_yaml_is_reported_with_its_position() -> None:
    # The missing "]" shows where the file ends.
    with pytest.raises(ConfigError, match=r"car\.yaml: not valid YAML \(line 3, column 1\)"):
        parse_config_text("vehicle:\n  mass: [1500\n", source="car.yaml")


def test_settings_must_come_in_sections() -> None:
    with pytest.raises(ConfigError, match=r"expected sections of settings .*, got \[1, 2\]"):
        parse_config_text("- 1\n- 2\n")


@pytest.mark.parametrize("encoding", ["utf-8", "utf-8-sig", "utf-16"])
def test_files_may_be_utf8_or_utf16(tmp_path: Path, encoding: str) -> None:
    # Windows PowerShell 5.1 writes UTF-16 with `racecar config > my.yaml`.
    path = tmp_path / "car.yaml"
    path.write_text("vehicle:\n  mass: 1500  # kg, ±50\n", encoding=encoding)

    assert read_config_file(path) == {"vehicle": {"mass": 1500}}


def test_a_file_that_is_not_text_is_reported(tmp_path: Path) -> None:
    path = tmp_path / "car.yaml"
    path.write_bytes(b"vehicle:\n  mass: \xff\n")

    with pytest.raises(ConfigError, match=r"car\.yaml: not readable text \(invalid start byte"):
        read_config_file(path)


def test_a_missing_file_is_reported(tmp_path: Path) -> None:
    with pytest.raises(ConfigError, match=r"missing\.yaml: can't read the file"):
        read_config_file(tmp_path / "missing.yaml")


# --------------------------------------------------------------------------- #
# Error messages
# --------------------------------------------------------------------------- #


def test_every_problem_is_listed_with_where_it_came_from(tmp_path: Path) -> None:
    first = write(tmp_path, "first.yaml", "vehicle:\n  mass: -5\n  width: 1.8\n")
    second = write(tmp_path, "second.yaml", "vehicle:\n  width: wide\n")

    assert problems([first, second], ["simulation.physics_hz=0"]) == [
        f'vehicle.width: must be a number, got "wide" (from {second})',
        f"vehicle.mass: must be greater than 0, got -5 (from {first})",
        "simulation.physics_hz: must be at least 1, got 0 (from --set)",
    ]


@pytest.mark.parametrize(
    ("override", "problem"),
    [
        ("vehicle.max_stear=3", "vehicle.max_stear: unknown setting; did you mean max_steer?"),
        ("vehicel.mass=3", "vehicel: unknown setting; did you mean vehicle?"),
        ("vehicle.colour=3", "vehicle.colour: unknown setting"),
        ("vehicle=fast", 'vehicle: must be a section of settings (key: value lines), got "fast"'),
        ("vehicle.mass=heavy", 'vehicle.mass: must be a number, got "heavy"'),
        ("vehicle.mass=", "vehicle.mass: must be a number, got nothing"),
        ("vehicle.mass=true", "vehicle.mass: must be a number, got true"),
        ("vehicle.mass=.nan", "vehicle.mass: must be a finite number, got NaN"),
        ("vehicle.mass=-.inf", "vehicle.mass: must be a finite number, got -Infinity"),
        ("simulation.physics_hz=60.0", "simulation.physics_hz: must be a whole number, got 60.0"),
        ("vehicle.max_steer=90", "vehicle.max_steer: must be less than 90, got 90"),
        ("vehicle.drag_coefficient=-0.1", "vehicle.drag_coefficient: must be at least 0, got -0.1"),
        ("race.off_track=fast", "race.off_track: must be one of 'none', 'slowdown', 'reset' or "
                                "'terminate', got \"fast\""),
        ("vehicle.wheelbase=5", "vehicle.wheelbase: must be shorter than the car's length (4.5 m), "
                                "got 5"),
        ("observation.rays=1", "observation.rays: must be true or false, got 1"),
    ],
)  # fmt: skip
def test_problems_are_explained_in_plain_words(override: str, problem: str) -> None:
    assert problems(overrides=[override]) == [f"{problem} (from --set)"]


def test_turning_off_every_input_says_so_without_repeating_them_all() -> None:
    overrides = [f"observation.{name}=false" for name in OBSERVATION_INPUTS]

    assert problems(overrides=overrides) == ["observation: turn on at least one input (from --set)"]


def test_setting_names_must_be_text(tmp_path: Path) -> None:
    path = write(tmp_path, "numbers.yaml", "vehicle:\n  3: 1500\n")

    assert problems([path]) == [f"vehicle.3: setting names must be text (from {path})"]


def test_a_problem_with_a_default_names_no_file() -> None:
    # The default wheelbase (2.7 m) doesn't fit a 2 m car; the problem shows on the wheelbase.
    assert problems(overrides=["vehicle.length=2"]) == [
        "vehicle.wheelbase: must be shorter than the car's length (2.0 m), got 2.7"
    ]


def test_a_problem_in_a_section_names_every_layer_that_wrote_to_it(tmp_path: Path) -> None:
    path = write(tmp_path, "car.yaml", "vehicel:\n  mass: 1500\n")

    assert problems([path], ["vehicel.width=2"]) == [
        f"vehicel: unknown setting; did you mean vehicle? (from {path}, --set)"
    ]


def test_a_problem_names_the_layer_that_last_set_the_value(tmp_path: Path) -> None:
    path = write(tmp_path, "car.yaml", "vehicle:\n  mass: -1\n")

    assert problems([path], ["vehicle=3", "vehicle.mass=-2"]) == [
        "vehicle.mass: must be greater than 0, got -2 (from --set)"
    ]


def test_values_json_cannot_show_are_still_shown() -> None:
    lines = problems(overrides=["vehicle.mass={2024-01-01: 1}"])

    assert lines == [
        "vehicle.mass: must be a number, got {datetime.date(2024, 1, 1): 1} (from --set)"
    ]


# --------------------------------------------------------------------------- #
# Writing
# --------------------------------------------------------------------------- #

lengths = st.floats(min_value=1e-3, max_value=1e3)
positive = st.floats(min_value=0, max_value=1e9, exclude_min=True)
non_negative = st.floats(min_value=0, max_value=1e9)
whole = st.integers(min_value=1, max_value=10**9)


@st.composite
def configs(draw: st.DrawFn) -> RacecarConfig:
    length = draw(lengths)
    vehicle = VehicleConfig(
        length=length,
        width=draw(positive),
        wheelbase=draw(st.floats(0, length, exclude_min=True, exclude_max=True)),
        mass=draw(positive),
        max_steer=draw(st.floats(0, 90, exclude_min=True, exclude_max=True)),
        steer_rate=draw(positive),
        grip=draw(positive),
        max_drive_force=draw(positive),
        max_power=draw(positive),
        max_brake_force=draw(positive),
        drag_coefficient=draw(non_negative),
        frontal_area=draw(positive),
        rolling_resistance=draw(non_negative),
    )
    simulation = SimulationConfig(physics_hz=draw(whole), action_repeat=draw(whole))
    race = RaceConfig(
        off_track=draw(st.sampled_from(["none", "slowdown", "reset", "terminate"])),
        grass_slowdown=draw(positive),
    )
    sensors = SensorConfig(
        rays=draw(whole),
        field_of_view=draw(st.floats(0, 360, exclude_min=True)),
        range=draw(positive),
    )
    inputs = draw(
        st.lists(
            st.booleans(), min_size=len(OBSERVATION_INPUTS), max_size=len(OBSERVATION_INPUTS)
        ).filter(any)
    )
    observation = ObservationConfig(
        **dict(zip(OBSERVATION_INPUTS, inputs, strict=True)),
        lookahead=draw(positive),
        lookahead_points=draw(whole),
    )
    reward = RewardConfig(**{term: draw(non_negative) for term in REWARD_TERMS})
    episode = EpisodeConfig(
        end_off_track=draw(st.booleans()),
        time_limit=draw(positive),
        stuck_time=draw(positive),
        start=draw(st.sampled_from(["grid", "random"])),
    )
    training = TrainingConfig(
        track=draw(st.sampled_from(["tracks/oval.json", "my track.json", "on"])),
        steps=draw(whole),
        cars=draw(whole),
        seed=draw(st.integers(0, 2**32)),
        device=draw(st.sampled_from(["cpu", "cuda", "auto"])),
        checkpoint_every=draw(whole),
        eval_every=draw(whole),
        eval_runs=draw(whole),
    )
    ppo = PPOConfig(
        learning_rate=draw(positive),
        steps_per_car=draw(st.integers(2, 10**6)),
        batch_size=draw(st.integers(2, 10**6)),
        epochs=draw(whole),
        gamma=draw(st.floats(0, 1, exclude_min=True)),
        gae_lambda=draw(st.floats(0, 1)),
        clip_range=draw(positive),
        entropy_coef=draw(non_negative),
        layers=draw(whole),
        layer_size=draw(whole),
    )
    return RacecarConfig(
        vehicle=vehicle,
        simulation=simulation,
        race=race,
        sensors=sensors,
        observation=observation,
        reward=reward,
        episode=episode,
        training=training,
        ppo=ppo,
    )


class Texts(BaseModel):
    plain: str = "slowdown"
    looks_like_yes: str = "on"
    looks_like_a_number: str = "1.5"
    empty: str = ""


def test_text_is_quoted_only_where_yaml_would_read_it_as_something_else() -> None:
    config = RacecarConfig.model_construct(vehicle=Texts())  # type: ignore[arg-type]

    text = format_config(config)

    lines = text.splitlines()
    for line in [
        "plain: slowdown",
        'looks_like_yes: "on"',
        'looks_like_a_number: "1.5"',
        'empty: ""',
    ]:
        assert f"  {line}" in lines
    assert parse_config_text(text)["vehicle"] == Texts().model_dump()


def test_a_setting_without_a_yaml_form_is_not_written() -> None:
    class Lists(BaseModel):
        items: list[int] = [1, 2]

    config = RacecarConfig.model_construct(vehicle=Lists())  # type: ignore[arg-type]

    with pytest.raises(TypeError, match="no YAML form for list settings"):
        format_config(config)


def test_on_off_settings_are_written_as_true_and_false() -> None:
    config = RacecarConfig(observation=ObservationConfig(rays=False))

    text = format_config(config)

    assert "  rays: false " in text
    assert "  speed: true " in text
    assert RacecarConfig.model_validate(parse_config_text(text)) == config


@given(configs())
def test_written_settings_read_back_exactly(config: RacecarConfig) -> None:
    text = format_config(config)

    assert RacecarConfig.model_validate(parse_config_text(text)) == config


def test_written_settings_explain_every_line() -> None:
    lines = format_config(RacecarConfig()).splitlines()

    settings = [line for line in lines if line.startswith("  ")]
    sections = (
        VehicleConfig,
        SimulationConfig,
        RaceConfig,
        SensorConfig,
        ObservationConfig,
        RewardConfig,
        EpisodeConfig,
        TrainingConfig,
        PPOConfig,
    )
    assert len(settings) == sum(len(section.model_fields) for section in sections)
    assert all("  # " in line for line in settings)
    assert all(len(line) <= 100 for line in lines)
    assert all(line.isascii() for line in lines)  # Windows consoles garble the rest

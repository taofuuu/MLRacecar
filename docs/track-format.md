# Track file format

> **In plain words:** A track is saved as a small text file listing the dots you placed and how
> wide the road is at each. The road's shape, edges, and checkpoints are worked out again every
> time the file is opened. Each file says which version of the format it uses, so files keep
> working as MLRacecar evolves.

Track files are JSON with the `.json` extension; the sample tracks live in
[`tracks/`](https://github.com/taofuuu/MLRacecar/tree/main/tracks). The reasoning behind the
format is in [ADR-0004](adr/0004-track-representation.md).

## Example

```json
{
  "schema_version": 1,
  "name": "Oval",
  "author": "MLRacecar",
  "description": "A 300 m x 160 m oval, 12 m wide, driven counter-clockwise.",
  "control_points": [
    {"x": 150.0, "y": 0.0, "width": 12.0},
    {"x": 138.6, "y": 30.6, "width": 12.0},
    {"x": 106.1, "y": 56.6, "width": 12.0}
  ],
  "metadata": {}
}
```

## Fields

| Field            | Type                      | Required | Meaning |
|------------------|---------------------------|----------|---------|
| `schema_version` | whole number              | yes      | Format version. This page describes version **1**. |
| `name`           | text, at least 1 character | yes     | The track's name. |
| `author`         | text                      | no       | Who made it. |
| `description`    | text                      | no       | Anything worth knowing about the track. |
| `control_points` | list of at least 3 points | yes      | The dots, in driving order. The first is on the start/finish line. |
| ↳ `x`, `y`       | number (metres)           | yes      | Position; x to the right, y up. |
| ↳ `width`        | number > 0 (metres)       | yes      | Road width at this dot; between dots it blends smoothly. |
| `metadata`       | JSON object               | no       | Free-form data for tools (an editor's view settings, for example). MLRacecar ignores it. |

Unknown fields are rejected, so a typo like `"widht"` is caught instead of silently ignored.
Numbers must be finite. Any precision is accepted, but the editor saves positions and widths
to the millimetre (3 decimal places) to keep files readable. The formal definition is the
JSON Schema: [`track-file-v1.json`](schemas/track-file-v1.json).

## Well-formed versus valid

Reading a file only checks that it is **well-formed**: the right fields, the right types,
positive widths, at least 3 points. Whether the **track** works (no folding edges, no crossings,
wide and long enough) is a separate step, the [track checks](reference.md). That way an unfinished
track can still be saved and reopened.

Check a file, both ways, from the command line:

```bash
uv run racecar check tracks/oval.json
```

## Versions and upgrades

- Every file records the `schema_version` it was written in.
- Older files are upgraded automatically when they are read, one version step at a time.
- A file from a **newer** MLRacecar is refused with a message saying to update, rather than
  half-read.
- Files are written with one control point per line, so in version control a diff shows exactly
  which point moved.

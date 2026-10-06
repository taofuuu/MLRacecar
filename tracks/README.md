# Sample tracks

| File | Track | Length |
|------|-------|--------|
| [`oval.json`](oval.json) | A 300 m x 160 m oval, 12 m wide: the simplest track to learn on | 739 m |
| [`gp-circuit.json`](gp-circuit.json) | A clockwise GP-style circuit with 13 corners at real-world sizes, 12-14 m wide | 3.5 km |

Both pass every track check with no issues; a test keeps it that way. Check any track with:

```bash
uv run racecar check tracks/oval.json
```

The file format is described in [docs/track-format.md](../docs/track-format.md).

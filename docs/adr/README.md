# Architecture Decision Records

> **In plain words:** Each file here records one big decision: what we picked, what else we
> considered, and why. Each one starts with a short plain-words summary; unfamiliar terms are
> explained in the [glossary](../glossary.md).

An ADR captures one significant decision: the context, the options we considered, what we
chose, and what it costs us. ADRs are immutable once accepted. To change a decision, write a
new ADR that supersedes the old one.

Lifecycle: `Proposed` → `Accepted` → (`Deprecated` | `Superseded by ADR-XXXX`)

To add one, copy [`0000-template.md`](0000-template.md), take the next number, and open a PR.

| ADR                                                  | Title                                                         | Status   |
|------------------------------------------------------|---------------------------------------------------------------|----------|
| [0001](0001-record-architecture-decisions.md)        | Record architecture decisions                                 | Accepted |
| [0002](0002-custom-2d-python-simulator.md)           | Build a custom 2D simulator in Python                         | Accepted |
| [0003](0003-layered-architecture-pure-core.md)       | Layered architecture with a pure NumPy simulation core        | Accepted |
| [0004](0004-track-representation.md)                 | Tracks as closed splines with a width profile, in versioned JSON | Accepted; curve type superseded by 0010 |
| [0005](0005-batched-multi-car-simulation.md)         | Batched struct-of-arrays simulation, multi-car from day one   | Accepted |
| [0006](0006-sb3-first-behind-agent-interface.md)     | Stable-Baselines3 first, behind an `Agent` interface          | Accepted |
| [0007](0007-python-toolchain.md)                     | Python toolchain: uv, ruff, mypy, pytest, GitHub Actions      | Accepted |
| [0008](0008-typed-configuration.md)                  | Typed configuration with pydantic and YAML                    | Accepted |
| [0009](0009-development-workflow.md)                 | Trunk-based workflow tracked in GitHub Issues and Projects    | Accepted |
| [0010](0010-c2-cubic-spline-centerline.md)           | Smooth-bend (C2) cubic spline for the track centerline        | Accepted |
| [0011](0011-immutable-editor-drafts.md)              | The editor edits immutable drafts                             | Accepted |

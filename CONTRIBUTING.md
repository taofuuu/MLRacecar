# Contributing

> **In plain words:** For every piece of work: take a ticket, make a separate branch, write the
> code and tests, then open a pull request that says which ticket it finishes. When the
> automatic checks pass, merge it. The ticket closes itself. Terms are in the
> [glossary](docs/glossary.md).

How work flows through this project, from idea to release. The reasoning is in
[ADR-0009](docs/adr/0009-development-workflow.md).

## Workflow

1. **Pick a ticket** from the *Ready* column of the [Project board](https://github.com/users/taofuuu/projects/3). If the ticket is
   `size:L`, consider splitting it first.
2. **Branch** off `main`: `<type>/<issue>-<short-slug>`, e.g. `feat/10-spline-centerline`.
3. **Commit** using [Conventional Commits](https://www.conventionalcommits.org/):
   `feat(track): add centripetal Catmull-Rom resampling`.
4. **Open a PR** early (draft is fine). Fill in the template and include `Closes #<issue>`.
5. **CI must be green.** Review, then **squash merge**. The linked issue moves to *Done*
   automatically.

## Continuous integration

Every PR and every push to `main` runs [`.github/workflows/ci.yml`](.github/workflows/ci.yml):

| Job                   | What it runs                                                            |
|-----------------------|-------------------------------------------------------------------------|
| Lint and type check   | All pre-commit hooks (the same ones that run on your commits)           |
| Tests (ubuntu-latest) | `pytest` with coverage; the coverage table appears in the run summary   |
| Tests (windows-latest)| The same, on Windows                                                    |

`main` is protected: changes land only through a PR, all three jobs must pass, and PRs are
squash-merged.

## Board columns

| Column          | Meaning                                                                       |
|-----------------|-------------------------------------------------------------------------------|
| **Backlog**     | Captured, not yet refined                                                     |
| **Ready**       | Meets the Definition of Ready, can be started                                 |
| **In Progress** | A branch exists; actively being worked on                                     |
| **In Review**   | PR open, waiting for CI and review                                            |
| **Done**        | Merged to `main` and meets the Definition of Done                             |

## Definition of Ready

- The goal is clear, with acceptance criteria written as testable statements.
- Dependencies are done, or explicitly not blocking.
- Size is S or M (L tickets get split).

## Definition of Done

- [ ] All acceptance criteria met
- [ ] Tests added or updated; coverage did not drop
- [ ] Lint, format, type check, and architecture rules pass
- [ ] Docs updated (user guide, architecture, or a new ADR if a decision was made)
- [ ] `CHANGELOG.md` updated under *Unreleased* for user-visible changes
- [ ] CI green on Windows and Linux

## Labels

| Prefix       | Values                                                                                           |
|--------------|--------------------------------------------------------------------------------------------------|
| `type:`      | `feature`, `chore`, `test`, `docs`, `research`, `bug`, `refactor`                                 |
| `area:`      | `infra`, `core`, `track`, `editor`, `sim`, `render`, `env`, `agents`, `training`, `web`          |
| `priority:`  | `P0` must-have for its milestone · `P1` should-have · `P2` nice-to-have                          |
| `size:`      | `S`, `M`, `L` (relative complexity, not hours)                                                   |

## Commit types

`feat` new capability · `fix` bug fix · `docs` documentation only · `test` tests only ·
`refactor` no behaviour change · `perf` performance · `chore` tooling/deps/CI ·
`build` packaging. Scope = area, e.g. `feat(env): …`.

## Local development

Prerequisites: Python 3.12 and [uv](https://docs.astral.sh/uv/getting-started/installation/).

```bash
uv sync                       # create .venv and install the project
uv run racecar --version      # check the install
```

Once after cloning, turn on the automatic checks for every commit:

```bash
uv run pre-commit install
```

Day-to-day commands:

| Command                                   | What it does                                                   |
|-------------------------------------------|----------------------------------------------------------------|
| `uv run pytest`                           | Fast tests + coverage report (fails below 80%)                 |
| `uv run pytest -m slow` / `-m gpu` / `-m benchmark` | Opt-in slow, GPU, or speed-measurement tests         |
| `uv run ruff check --fix .`               | Lint, auto-fixing what it safely can                           |
| `uv run ruff format .`                    | Format all code                                                |
| `uv run mypy`                             | Strict type check of `src/`, `tests/`, `scripts/`              |
| `uv run pre-commit run --all-files`       | Everything the commit hook runs, on the whole repo             |

Coming later in M0: `uv run lint-imports` (architecture rules, #6).

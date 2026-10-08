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
   `feat(track): add arc-length resampling to the spline`.
4. **Open a PR** early (draft is fine). The PR template asks for a summary, a plain-words
   box, how each acceptance criterion was verified, and the Definition of Done. Keep
   `Closes #<issue>` at the bottom.
5. **CI must be green.** Review, then **squash merge**. The linked issue moves to *Done*
   automatically.

## Creating a ticket

New issues go through one of three forms (blank issues are turned off):

| Form               | Use it for                                                    | Gets the label  |
|--------------------|---------------------------------------------------------------|-----------------|
| **Feature**        | A new capability                                              | `type:feature`  |
| **Bug**            | Something behaves differently from the docs or tests          | `type:bug`      |
| **Research spike** | A time-boxed question that ends in an ADR or experiment log   | `type:research` |

Every form adds the issue to the Project board, where it lands in **Backlog**. To move it to
**Ready**, add its `area:`, `priority:`, and `size:` labels and check the Definition of Ready.

## Continuous integration

Every PR and every push to `main` runs [`.github/workflows/ci.yml`](.github/workflows/ci.yml):

| Job                   | What it runs                                                            |
|-----------------------|-------------------------------------------------------------------------|
| Lint and type check   | All pre-commit hooks (the same ones that run on your commits)           |
| Tests (ubuntu-latest) | `pytest` with coverage; the coverage table appears in the run summary   |
| Tests (windows-latest)| The same, on Windows                                                    |
| Build docs            | `mkdocs build --strict`: broken links or anchors fail ([docs.yml](.github/workflows/docs.yml)); on `main` it also publishes the site |

`main` is protected: changes land only through a PR, all four jobs must pass, and PRs are
squash-merged.

## Board columns

| Column          | Meaning                                              | How a card gets there              |
|-----------------|------------------------------------------------------|------------------------------------|
| **Backlog**     | Captured, not yet refined                            | Automatic when an issue is created |
| **Ready**       | Meets the Definition of Ready, can be started        | By hand, during refinement         |
| **In Progress** | A branch exists; actively being worked on            | By hand, when work starts          |
| **In Review**   | PR open, waiting for CI and review                   | Automatic when a PR links the issue |
| **Done**        | Merged to `main` and meets the Definition of Done    | Automatic when the PR is merged    |

Board views: **Board** (cards by column), **By milestone** (table grouped by milestone), and
**Current milestone** (the board, filtered to the active milestone).

## Definition of Ready

- The goal is clear, with acceptance criteria written as testable statements.
- Dependencies are done, or explicitly not blocking.
- It has `area:`, `priority:`, and `size:` labels.
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
| `uv run pytest -m slow --no-cov` / `-m gpu` | Opt-in slow or GPU tests (without coverage, whose 80% minimum fails when only a few tests run) |
| `uv run pytest -m benchmark --no-cov`     | Speed measurements (without coverage, which would slow down what is timed and fail the 80% minimum) |
| `uv run python scripts/benchmark_table.py benchmark.json` | The README's speed table, from results saved with `--benchmark-json=benchmark.json` |
| `uv run ruff check --fix .`               | Lint, auto-fixing what it safely can                           |
| `uv run ruff format .`                    | Format all code                                                |
| `uv run mypy`                             | Strict type check of `src/`, `tests/`, `scripts/`              |
| `uv run racecar check tracks/oval.json`  | Read a track file and run the track checks on it               |
| `uv run racecar drive tracks/oval.json`  | Drive the track with the keyboard (needs a display)            |
| `uv run python scripts/export_track_schema.py` | Re-publish the track file JSON Schema after changing the format |
| `uv run python scripts/export_default_config.py` | Rewrite `configs/default.yaml` after adding or changing a setting |
| `uv run python scripts/update_golden.py` | Record the golden trajectories again after an intended physics change (say why in the PR) |
| `uv run lint-imports`                     | Architecture check: each layer only imports the layers below it |
| `uv run --group docs mkdocs serve`        | Preview the docs site at http://127.0.0.1:8000 while editing   |
| `uv run pre-commit run --all-files`       | Everything the commit hook runs, on the whole repo             |

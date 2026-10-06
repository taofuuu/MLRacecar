# ADR-0009: Trunk-based workflow tracked in GitHub Issues and Projects

- **Status:** Accepted
- **Date:** 2026-10-06
- **Related:** [CONTRIBUTING.md](../../CONTRIBUTING.md), ticket M0-7

> **In plain words:** Every piece of work is a ticket on GitHub. You do it on a short-lived branch, merge it through a pull request that names the ticket, and the ticket closes and the board updates by itself. Commit messages follow one standard format.

## Context

The backlog must show what is planned, in progress, and done, and the history must show a
professional engineering process to anyone browsing the repository.

## Decision

- **Backlog:** GitHub Issues, with labels (`type:*`, `area:*`, `priority:*`, `size:*`) and
  milestones (M0–M8), on a GitHub Project board:
  `Backlog → Ready → In Progress → In Review → Done`.
- **Branching:** trunk-based. Short-lived branches off `main` named `<type>/<issue>-<slug>`
  (e.g. `feat/10-spline-centerline`), merged by squash PR. `main` is protected and requires
  green CI.
- **Commits:** [Conventional Commits](https://www.conventionalcommits.org/)
  (`feat:`, `fix:`, `docs:`, `test:`, `chore:`, `refactor:`, `perf:`).
- **Every PR closes an issue** (`Closes #N`), so the board updates automatically.
- **Releases:** SemVer tags per milestone (`v0.1.0` = MVP), with a GitHub Release and
  `CHANGELOG.md` entry.

## Consequences

- **Positive:** progress is visible and traceable from idea → ticket → PR → release.
  Mirrors how teams work.
- **Negative / costs:** some ceremony for a solo project, kept light with templates and
  automation.

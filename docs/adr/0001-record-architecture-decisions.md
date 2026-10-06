# ADR-0001: Record architecture decisions

- **Status:** Accepted
- **Date:** 2026-10-06

> **In plain words:** We write down every big decision in a short file: what we chose, what else we considered, and why. Anyone (including future you) can then see the reasoning, not just the code.

## Context

This project is meant to be read by other engineers: reviewers, interviewers, future
contributors. Code shows *what* we built; it rarely shows *why*, or which alternatives we
rejected. That reasoning gets lost unless we write it down when we decide.

## Options considered

1. **No formal record:** decisions live in commit messages and memory. Cheap, but they get lost.
2. **One big design doc:** gets long and stale, and its history is hard to follow.
3. **Architecture Decision Records:** short, numbered, immutable files, one per decision.

## Decision

We use lightweight ADRs (a MADR-style template) in `docs/adr/`. An ADR is required when a
decision is hard to reverse, affects more than one package, or picks between credible
alternatives.

## Consequences

- **Positive:** the "why" is preserved and reviewable in PRs, and it makes good interview
  material.
- **Negative / costs:** some writing overhead per decision, kept small by the template.

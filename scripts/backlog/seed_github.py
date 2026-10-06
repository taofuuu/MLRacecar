"""Seed the GitHub backlog (labels, milestones, issues) from backlog.toml.

The script is idempotent: labels are created or updated, while milestones and
issues that already exist (matched by title) are left untouched. Re-running it
after adding entries to backlog.toml only creates the new ones.

Requires the GitHub CLI (`gh`), authenticated. Adding issues to a Project board
additionally needs the `project` scope: `gh auth refresh -s project`.

Usage:
    python scripts/backlog/seed_github.py --dry-run
    python scripts/backlog/seed_github.py --repo OWNER/NAME
    python scripts/backlog/seed_github.py --repo OWNER/NAME --project 1
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tomllib
from pathlib import Path
from typing import Any

BACKLOG_FILE = Path(__file__).with_name("backlog.toml")
VALID_STATUSES = {"open", "done"}


def load_backlog(path: Path) -> dict[str, Any]:
    with path.open("rb") as f:
        return tomllib.load(f)


def validate(backlog: dict[str, Any]) -> list[str]:
    """Return a list of problems; an empty list means the backlog is consistent."""
    problems: list[str] = []
    labels = {label["name"] for label in backlog.get("labels", [])}
    milestones = {m["key"] for m in backlog.get("milestones", [])}
    seen_keys: set[str] = set()
    seen_titles: set[str] = set()

    for issue in backlog.get("issues", []):
        key = issue.get("key", "<missing key>")
        if key in seen_keys:
            problems.append(f"{key}: duplicate key")
        if issue["title"] in seen_titles:
            problems.append(f"{key}: duplicate title {issue['title']!r}")
        if issue.get("milestone") not in milestones:
            problems.append(f"{key}: unknown milestone {issue.get('milestone')!r}")
        problems.extend(
            f"{key}: unknown label {label!r}" for label in issue.get("labels", []) if label not in labels
        )
        # Dependencies must point to earlier entries so issue numbers are known at creation time.
        problems.extend(
            f"{key}: depends on {dep!r}, which is not defined earlier in the file"
            for dep in issue.get("depends_on", [])
            if dep not in seen_keys
        )
        if issue.get("status", "open") not in VALID_STATUSES:
            problems.append(f"{key}: status must be one of {sorted(VALID_STATUSES)}")
        seen_keys.add(key)
        seen_titles.add(issue["title"])
    return problems


def render_body(issue: dict[str, Any], refs: dict[str, str]) -> str:
    lines = [issue["summary"].strip(), ""]
    if tasks := issue.get("tasks"):
        lines += ["## Tasks", *(f"- [ ] {task}" for task in tasks), ""]
    if acceptance := issue.get("acceptance"):
        lines += ["## Acceptance criteria", *(f"- [ ] {item}" for item in acceptance), ""]
    if deps := issue.get("depends_on"):
        lines += ["## Depends on", *(f"- {refs[dep]}" for dep in deps), ""]
    lines.append(f"<sub>Backlog key: `{issue['key']}`</sub>")
    return "\n".join(lines)


class GitHub:
    def __init__(self, repo: str, *, dry_run: bool) -> None:
        self.repo = repo
        self.dry_run = dry_run

    def run(self, *args: str, stdin: str | None = None, mutates: bool = True) -> str:
        if self.dry_run and mutates:
            print(f"  [dry-run] gh {' '.join(args)}")
            return ""
        result = subprocess.run(
            ["gh", *args],
            input=stdin,
            capture_output=True,
            text=True,
            encoding="utf-8",
            check=False,
        )
        if result.returncode != 0:
            raise RuntimeError(f"gh {' '.join(args)} failed:\n{result.stderr.strip()}")
        return result.stdout.strip()

    def existing_milestones(self) -> set[str]:
        if self.dry_run:
            return set()
        out = self.run(
            "api", f"repos/{self.repo}/milestones?state=all&per_page=100", "--paginate", mutates=False
        )
        return {m["title"] for m in json.loads(out or "[]")}

    def existing_issues(self) -> dict[str, int]:
        if self.dry_run:
            return {}
        out = self.run(
            "issue", "list", "--repo", self.repo, "--state", "all", "--limit", "1000",
            "--json", "number,title", mutates=False,
        )  # fmt: skip
        return {i["title"]: i["number"] for i in json.loads(out or "[]")}


def seed(backlog: dict[str, Any], gh: GitHub, project: int | None) -> None:
    owner = gh.repo.split("/")[0]
    milestone_titles = {m["key"]: m["title"] for m in backlog["milestones"]}

    print("Labels")
    for label in backlog["labels"]:
        gh.run(
            "label", "create", label["name"], "--repo", gh.repo, "--color", label["color"],
            "--description", label["description"], "--force",
        )  # fmt: skip

    print("Milestones")
    existing_ms = gh.existing_milestones()
    for m in backlog["milestones"]:
        if m["title"] in existing_ms:
            print(f"  exists: {m['title']}")
            continue
        args = ["api", f"repos/{gh.repo}/milestones", "-f", f"title={m['title']}",
                "-f", f"description={m['description']}"]  # fmt: skip
        if due := m.get("due"):
            args += ["-f", f"due_on={due}T23:59:59Z"]
        gh.run(*args)
        print(f"  created: {m['title']}")

    print("Issues")
    existing = gh.existing_issues()
    refs: dict[str, str] = {}
    for issue in backlog["issues"]:
        title = issue["title"]
        if title in existing:
            number = existing[title]
            print(f"  exists: #{number} {title}")
        else:
            args = ["issue", "create", "--repo", gh.repo, "--title", title, "--body-file", "-",
                    "--milestone", milestone_titles[issue["milestone"]]]  # fmt: skip
            for label in issue.get("labels", []):
                args += ["--label", label]
            url = gh.run(*args, stdin=render_body(issue, refs))
            number = int(url.rstrip("/").rsplit("/", 1)[-1]) if url else 0
            print(f"  created: #{number} {title}" if url else f"  would create: {issue['key']} {title}")
            if issue.get("status") == "done" and url:
                gh.run("issue", "close", str(number), "--repo", gh.repo, "--reason", "completed")
            if project is not None and url:
                gh.run("project", "item-add", str(project), "--owner", owner, "--url", url)
        refs[issue["key"]] = f"#{number}" if number else f"{issue['key']} ({title})"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--repo", default="OWNER/NAME", help="GitHub repository, e.g. octocat/MLRacecar")
    parser.add_argument("--project", type=int, help="Project number to add newly created issues to")
    parser.add_argument("--file", type=Path, default=BACKLOG_FILE, help="Backlog TOML file")
    parser.add_argument("--dry-run", action="store_true", help="Validate and print actions without calling GitHub")
    args = parser.parse_args()

    backlog = load_backlog(args.file)
    if problems := validate(backlog):
        print("Backlog is invalid:", *problems, sep="\n  ", file=sys.stderr)
        return 1
    issues = backlog["issues"]
    print(f"Backlog OK: {len(backlog['labels'])} labels, {len(backlog['milestones'])} milestones, {len(issues)} issues")

    if not args.dry_run and args.repo == "OWNER/NAME":
        parser.error("--repo is required unless --dry-run is given")
    seed(backlog, GitHub(args.repo, dry_run=args.dry_run), args.project)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

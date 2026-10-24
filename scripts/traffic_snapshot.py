#!/usr/bin/env python3
"""Daily traction snapshot for the JARVIS repository.

GitHub keeps repository traffic (views, clones, referrers, popular pages) for
only 14 days, so it has to be archived daily or it is lost. This script pulls
everything that signals traction and merges it into a small on-disk history:

  * traffic views + clones per day      (needs a token with Administration: read)
  * top referrers + popular paths        (same token; rolling 14-day window)
  * stars / forks / watchers / issues    (public)
  * release asset downloads              (public)
  * Home Assistant analytics installs    (public; opted-in HA instances only)
  * the HA community forum thread        (public: views, replies, who's waiting)

It then renders REPORT.md — the page the growth playbook (docs/GROWTH.md) is
driven from. Stdlib only, so the workflow needs no dependencies.

Usage:  python3 scripts/traffic_snapshot.py <data_dir> [--repo owner/name]
Token:  TRAFFIC_TOKEN (preferred) or GITHUB_TOKEN from the environment.
"""
from __future__ import annotations

import argparse
import datetime as dt
import html
import json
import os
import pathlib
import re
import sys
import urllib.error
import urllib.request

API = "https://api.github.com"
HA_ANALYTICS = "https://analytics.home-assistant.io/custom_integrations.json"
DOMAIN = "jarvis"
# The JARVIS thread on community.home-assistant.io ("Share your Projects!").
FORUM = "https://community.home-assistant.io"
FORUM_TOPIC = os.environ.get("JARVIS_FORUM_TOPIC", "1027117")


# ── fetching ─────────────────────────────────────────────────────────────────
def _get(url: str, token: str | None = None, accept: str = "application/vnd.github+json"):
    req = urllib.request.Request(url, headers={"Accept": accept, "User-Agent": "jarvis-traffic"})
    if token:
        req.add_header("Authorization", f"Bearer {token}")
    with urllib.request.urlopen(req, timeout=30) as resp:
        return json.load(resp)


def fetch(repo: str, token: str | None) -> dict:
    """Return one raw snapshot. Failing sources are recorded, never fatal."""
    snap: dict = {"errors": {}}
    base = f"{API}/repos/{repo}"

    def grab(key: str, url: str, **kw):
        try:
            snap[key] = _get(url, **kw)
        except (urllib.error.URLError, TimeoutError, ValueError) as err:
            snap["errors"][key] = str(err)

    grab("repo", base, token=token)
    grab("releases", f"{base}/releases?per_page=100", token=token)
    for key in ("views", "clones"):
        grab(key, f"{base}/traffic/{key}", token=token)
    grab("referrers", f"{base}/traffic/popular/referrers", token=token)
    grab("paths", f"{base}/traffic/popular/paths", token=token)
    grab("ha_analytics", HA_ANALYTICS)
    if FORUM_TOPIC:
        # print=true returns the whole thread (Discourse caps it at 1000 posts).
        grab("forum", f"{FORUM}/t/{FORUM_TOPIC}.json?print=true", accept="application/json")
    return snap


# ── merging ──────────────────────────────────────────────────────────────────
def merge_daily(history: dict, series: dict | None, key: str) -> None:
    """Fold a 14-day traffic series into history[date][key].

    The newest API figure for a day wins: today's partial count is overwritten
    by tomorrow's complete one, while days older than the window are kept.
    """
    for row in (series or {}).get(key, []):
        day = row["timestamp"][:10]
        history.setdefault(day, {})[key] = {"count": row["count"], "uniques": row["uniques"]}


def summarize(snap: dict, today: str) -> dict:
    """Reduce a raw snapshot to the per-day point stored in snapshots.jsonl."""
    repo = snap.get("repo") or {}
    releases = snap.get("releases") or []
    ha = (snap.get("ha_analytics") or {}).get(DOMAIN) or {}
    return {
        "date": today,
        "stars": repo.get("stargazers_count"),
        "forks": repo.get("forks_count"),
        "watchers": repo.get("subscribers_count"),
        "open_issues": repo.get("open_issues_count"),
        "release_downloads": sum(a.get("download_count", 0) for r in releases for a in r.get("assets", [])),
        "ha_installs": ha.get("total"),
        "referrers": [
            {"referrer": r["referrer"], "count": r["count"], "uniques": r["uniques"]}
            for r in (snap.get("referrers") or [])
        ],
        "paths": [
            {"path": p["path"], "count": p["count"], "uniques": p["uniques"]}
            for p in (snap.get("paths") or [])
        ][:10],
        "forum": summarize_forum(snap.get("forum"), today),
        "errors": sorted(snap.get("errors", {})),
    }


def _excerpt(cooked: str, limit: int = 200) -> str:
    text = html.unescape(re.sub(r"<[^>]+>", " ", cooked or ""))
    text = " ".join(text.split())
    return text if len(text) <= limit else text[: limit - 1].rstrip() + "…"


def summarize_forum(topic: dict | None, today: str, days: int = 7) -> dict | None:
    """The forum thread's numbers plus the posts from the last `days` days.

    `waiting` is true when the newest post isn't by the thread's author, i.e.
    someone in the thread is waiting on a reply.
    """
    if not topic:
        return None
    posts = (topic.get("post_stream") or {}).get("posts") or []
    op = ((topic.get("details") or {}).get("created_by") or {}).get("username") \
        or (posts[0].get("username") if posts else None)
    since = (dt.date.fromisoformat(today) - dt.timedelta(days=days)).isoformat()
    recent = [
        {"n": p.get("post_number"), "user": p.get("username"),
         "date": (p.get("created_at") or "")[:10], "excerpt": _excerpt(p.get("cooked", ""))}
        for p in posts
        if (p.get("created_at") or "")[:10] >= since and p.get("post_number") != 1
    ]
    last = posts[-1] if posts else {}
    return {
        "title": topic.get("title"),
        "url": f"{FORUM}/t/{topic.get('slug', 'topic')}/{topic.get('id', FORUM_TOPIC)}",
        "views": topic.get("views"),
        "replies": max((topic.get("posts_count") or 1) - 1, 0),
        "author": op,
        "last_poster": last.get("username"),
        "last_posted_at": (last.get("created_at") or "")[:10],
        "waiting": bool(last) and last.get("username") != op,
        "recent": recent[-10:],
    }


def upsert_point(points: list[dict], point: dict) -> list[dict]:
    """Replace any same-day point (re-runs are idempotent), keep date order."""
    kept = [p for p in points if p["date"] != point["date"]]
    return sorted(kept + [point], key=lambda p: p["date"])


# ── reporting ────────────────────────────────────────────────────────────────
def _window(daily: dict, key: str, end: dt.date, days: int) -> tuple[int, int]:
    count = uniques = 0
    for i in range(days):
        d = (end - dt.timedelta(days=i)).isoformat()
        row = daily.get(d, {}).get(key)
        if row:
            count += row["count"]
            uniques += row["uniques"]
    return count, uniques


def _delta(now: int, before: int) -> str:
    if not before:
        return "n/a" if not now else "new"
    return f"{(now - before) / before:+.0%}"


def _first_on_or_after(points: list[dict], day: str) -> dict | None:
    return next((p for p in points if p["date"] >= day), None)


def _forum_section(forum: dict | None, points: list[dict], today: dt.date) -> list[str]:
    out = ["", "## HA forum thread", ""]
    if not forum:
        return out + ["_Forum thread not reachable today._"]
    then = _first_on_or_after(points, (today - dt.timedelta(days=7)).isoformat())
    prev = {} if not then or then.get("date") == today.isoformat() else (then.get("forum") or {})

    def wk(field: str) -> str:
        if not prev or prev.get(field) is None or forum.get(field) is None:
            return ""
        return f" ({forum[field] - prev[field]:+d} this week)"

    out.append(f"[{forum.get('title') or 'Thread'}]({forum['url']}): "
               f"**{forum.get('views')}** views{wk('views')}, "
               f"**{forum.get('replies')}** replies{wk('replies')}.")
    if forum.get("waiting"):
        out += ["", f"> **Waiting on a reply:** the newest post is from "
                    f"@{forum.get('last_poster')} ({forum.get('last_posted_at')})."]
    recent = forum.get("recent") or []
    out += ["", "New posts in the last 7 days:" if recent else "_No new posts in the last 7 days._"]
    out += [f"- #{p['n']} @{p['user']} ({p['date']}): {p['excerpt']}" for p in recent]
    return out


def render_report(daily: dict, points: list[dict], today: dt.date) -> str:
    latest = points[-1] if points else {}
    # Traffic for "today" is partial, and GitHub can lag several days behind.
    # Anchor on the newest complete day it has reported, so missing days are
    # not mistaken for a drop.
    end = today - dt.timedelta(days=1)
    reported = [d for d in daily if d <= end.isoformat()]
    if reported:
        end = dt.date.fromisoformat(max(reported))
    v7, vu7 = _window(daily, "views", end, 7)
    v7p, vu7p = _window(daily, "views", end - dt.timedelta(days=7), 7)
    c7, cu7 = _window(daily, "clones", end, 7)
    c7p, cu7p = _window(daily, "clones", end - dt.timedelta(days=7), 7)

    def growth(field: str, days: int) -> str:
        then = _first_on_or_after(points, (today - dt.timedelta(days=days)).isoformat())
        if not then or then is latest or then.get(field) is None or latest.get(field) is None:
            return "—"
        return f"{latest[field] - then[field]:+d}"

    out = [
        f"# JARVIS traction report — {today.isoformat()}",
        "",
        "Generated daily by `.github/workflows/traffic.yml`. What to do with these",
        "numbers lives in `docs/GROWTH.md` on the default branch. Clone counts",
        "include CI checkouts, so treat views and referrers as the audience signal.",
        "",
        f"## 7 days to {end.isoformat()} vs the 7 before",
        "",
        "| Metric | Last 7d | Prior 7d | Change |",
        "|---|---:|---:|---:|",
        f"| Views | {v7} | {v7p} | {_delta(v7, v7p)} |",
        f"| Unique visitors (sum of daily) | {vu7} | {vu7p} | {_delta(vu7, vu7p)} |",
        f"| Clones | {c7} | {c7p} | {_delta(c7, c7p)} |",
        f"| Unique cloners (sum of daily) | {cu7} | {cu7p} | {_delta(cu7, cu7p)} |",
        "",
        "## Totals",
        "",
        "| Metric | Now | 7d change | 30d change |",
        "|---|---:|---:|---:|",
    ]
    for label, field in (
        ("Stars", "stars"),
        ("Forks", "forks"),
        ("Watchers", "watchers"),
        ("HA analytics installs", "ha_installs"),
        ("Release asset downloads", "release_downloads"),
    ):
        val = latest.get(field)
        out.append(f"| {label} | {'—' if val is None else val} | {growth(field, 7)} | {growth(field, 30)} |")

    out += _forum_section(latest.get("forum"), points, today)

    out += ["", "## Top referrers (rolling 14 days)", ""]
    refs = latest.get("referrers") or []
    if refs:
        out += ["| Source | Views | Uniques |", "|---|---:|---:|"]
        out += [f"| {r['referrer']} | {r['count']} | {r['uniques']} |" for r in refs]
    else:
        out.append("_No referrer data (missing TRAFFIC_TOKEN, or no traffic yet)._")

    out += ["", "## Most-visited pages (rolling 14 days)", ""]
    paths = latest.get("paths") or []
    if paths:
        out += ["| Path | Views | Uniques |", "|---|---:|---:|"]
        out += [f"| `{p['path']}` | {p['count']} | {p['uniques']} |" for p in paths]
    else:
        out.append("_No page data._")

    out += ["", "## Daily views & clones (last 30 days)", "",
            "| Date | Views | Uniq | Clones | Uniq |", "|---|---:|---:|---:|---:|"]
    for i in range(30):
        d = (today - dt.timedelta(days=i)).isoformat()
        row = daily.get(d)
        if not row:
            continue
        v = row.get("views", {})
        c = row.get("clones", {})
        out.append(f"| {d} | {v.get('count', 0)} | {v.get('uniques', 0)} | {c.get('count', 0)} | {c.get('uniques', 0)} |")

    if latest.get("errors"):
        out += ["", f"> Sources that failed today: {', '.join(latest['errors'])}"]
    return "\n".join(out) + "\n"


# ── entry point ──────────────────────────────────────────────────────────────
def run(data_dir: pathlib.Path, snap: dict, today: dt.date) -> str:
    data_dir.mkdir(parents=True, exist_ok=True)
    daily_path = data_dir / "daily.json"
    points_path = data_dir / "snapshots.jsonl"

    daily = json.loads(daily_path.read_text()) if daily_path.exists() else {}
    merge_daily(daily, snap.get("views"), "views")
    merge_daily(daily, snap.get("clones"), "clones")
    daily_path.write_text(json.dumps(dict(sorted(daily.items())), indent=1) + "\n")

    points = [json.loads(line) for line in points_path.read_text().splitlines() if line.strip()] \
        if points_path.exists() else []
    points = upsert_point(points, summarize(snap, today.isoformat()))
    points_path.write_text("".join(json.dumps(p) + "\n" for p in points))

    report = render_report(daily, points, today)
    (data_dir / "REPORT.md").write_text(report)
    return report


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("data_dir", type=pathlib.Path)
    ap.add_argument("--repo", default=os.environ.get("GITHUB_REPOSITORY", "sam3gp8/jarvis-aio"))
    args = ap.parse_args()

    token = os.environ.get("TRAFFIC_TOKEN") or os.environ.get("GITHUB_TOKEN")
    snap = fetch(args.repo, token)
    for key, err in snap["errors"].items():
        print(f"warning: {key}: {err}", file=sys.stderr)
    print(run(args.data_dir, snap, dt.datetime.now(dt.timezone.utc).date()))
    return 0


if __name__ == "__main__":
    sys.exit(main())

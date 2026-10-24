"""Traction snapshot: history merging and report rendering (no network)."""
from __future__ import annotations

import datetime as dt
import importlib.util
import json
import pathlib

_PATH = pathlib.Path(__file__).resolve().parent.parent / "scripts" / "traffic_snapshot.py"
_spec = importlib.util.spec_from_file_location("traffic_snapshot", _PATH)
ts = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(ts)


def _series(key, rows):
    return {key: [{"timestamp": f"{d}T00:00:00Z", "count": c, "uniques": u} for d, c, u in rows]}


def _snap(views, stars=10, referrers=()):
    return {
        "repo": {"stargazers_count": stars, "forks_count": 1, "subscribers_count": 2, "open_issues_count": 0},
        "releases": [{"assets": [{"download_count": 4}]}],
        "views": _series("views", views),
        "clones": _series("clones", [(d, 1, 1) for d, _, _ in views]),
        "referrers": [{"referrer": r, "count": 5, "uniques": 3} for r in referrers],
        "paths": [],
        "ha_analytics": {"jarvis": {"total": 42}},
        "errors": {},
    }


def test_history_survives_the_14_day_window_and_newest_count_wins(tmp_path):
    ts.run(tmp_path, _snap([("2026-09-01", 3, 2), ("2026-09-02", 1, 1)]), dt.date(2026, 9, 2))
    # Next day: 09-01 has aged out of the API window, 09-02 is now complete.
    ts.run(tmp_path, _snap([("2026-09-02", 7, 4), ("2026-09-03", 2, 2)]), dt.date(2026, 9, 3))

    daily = json.loads((tmp_path / "daily.json").read_text())
    assert daily["2026-09-01"]["views"] == {"count": 3, "uniques": 2}
    assert daily["2026-09-02"]["views"] == {"count": 7, "uniques": 4}
    assert list(daily) == sorted(daily)


def test_same_day_rerun_is_idempotent(tmp_path):
    ts.run(tmp_path, _snap([("2026-09-02", 1, 1)], stars=10), dt.date(2026, 9, 2))
    ts.run(tmp_path, _snap([("2026-09-02", 1, 1)], stars=11), dt.date(2026, 9, 2))
    points = (tmp_path / "snapshots.jsonl").read_text().splitlines()
    assert len(points) == 1
    assert json.loads(points[0])["stars"] == 11


def test_report_compares_weeks_and_lists_referrers(tmp_path):
    today = dt.date(2026, 9, 20)
    views = [((today - dt.timedelta(days=i)).isoformat(), 10 if i <= 7 else 5, 2) for i in range(1, 15)]
    ts.run(tmp_path, _snap(views, stars=10), today - dt.timedelta(days=7))
    report = ts.run(tmp_path, _snap(views, stars=15, referrers=["reddit.com"]), today)

    assert "| Views | 70 | 35 | +100% |" in report
    assert "| Stars | 15 | +5 |" in report
    assert "| HA analytics installs | 42 |" in report
    assert "| reddit.com | 5 | 3 |" in report


def test_missing_token_degrades_to_public_metrics(tmp_path):
    snap = _snap([])
    for key in ("views", "clones", "referrers", "paths"):
        snap.pop(key)
    snap["errors"] = {"views": "HTTP Error 403"}
    report = ts.run(tmp_path, snap, dt.date(2026, 9, 2))
    assert "No referrer data" in report
    assert "Sources that failed today: views" in report
    assert "| Stars | 10 |" in report


def test_week_windows_anchor_on_latest_reported_day(tmp_path):
    # GitHub stopped reporting 4 days ago; that gap must not read as a drop.
    today = dt.date(2026, 9, 28)
    last = today - dt.timedelta(days=5)
    views = [((last - dt.timedelta(days=i)).isoformat(), 10, 2) for i in range(14)]
    report = ts.run(tmp_path, _snap(views), today)
    assert f"## 7 days to {last.isoformat()} vs the 7 before" in report
    assert "| Views | 70 | 70 | +0% |" in report


def _topic(posts, views=900):
    return {
        "id": 1027117, "slug": "jarvis-aio", "title": "Jarvis-AIO", "views": views,
        "posts_count": len(posts), "details": {"created_by": {"username": "sam"}},
        "post_stream": {"posts": [
            {"post_number": i + 1, "username": u, "created_at": f"{d}T10:00:00Z",
             "cooked": f"<p>{t} &amp; more</p>"}
            for i, (u, d, t) in enumerate(posts)
        ]},
    }


def test_forum_thread_flags_unanswered_post(tmp_path):
    snap = _snap([])
    snap["forum"] = _topic([("sam", "2026-10-02", "launch"),
                            ("william", "2026-10-23", "cant get email working")])
    report = ts.run(tmp_path, snap, dt.date(2026, 10, 24))
    assert "**900** views, **1** replies." in report
    assert "Waiting on a reply:** the newest post is from @william (2026-10-23)" in report
    assert "- #2 @william (2026-10-23): cant get email working & more" in report
    assert "launch" not in report.split("New posts")[1]   # the opening post isn't "new"


def test_forum_thread_weekly_change_and_answered(tmp_path):
    first = _snap([])
    first["forum"] = _topic([("sam", "2026-10-02", "launch")], views=800)
    ts.run(tmp_path, first, dt.date(2026, 10, 17))
    later = _snap([])
    later["forum"] = _topic([("sam", "2026-10-02", "launch"),
                             ("william", "2026-10-20", "q"), ("sam", "2026-10-21", "a")], views=950)
    report = ts.run(tmp_path, later, dt.date(2026, 10, 24))
    assert "**950** views (+150 this week), **2** replies (+2 this week)." in report
    assert "Waiting on a reply" not in report


def test_forum_unreachable_is_reported(tmp_path):
    report = ts.run(tmp_path, _snap([]), dt.date(2026, 10, 24))
    assert "Forum thread not reachable today" in report

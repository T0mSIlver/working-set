"""scripts/pi_concurrency.py: how many pi sessions are in use at once.

Session files are made up: generic names, times in minutes from one start.
"""
from __future__ import annotations

import importlib.util
import json
import shutil
import subprocess
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parent.parent / "scripts" / "pi_concurrency.py"
spec = importlib.util.spec_from_file_location("pi_concurrency", SCRIPT)
pc = importlib.util.module_from_spec(spec)
spec.loader.exec_module(pc)

T0 = datetime(2026, 3, 10, 12, 0, tzinfo=timezone.utc)


@pytest.fixture(autouse=True)
def utc(monkeypatch):
    """Days are the machine's local days: pin them."""
    if hasattr(time, "tzset"):
        monkeypatch.setenv("TZ", "UTC")
        time.tzset()
        yield
        monkeypatch.undo()
        time.tzset()
    else:
        yield


def iso(minutes, day=0):
    t = T0 + timedelta(days=day, minutes=minutes)
    return t.isoformat(timespec="milliseconds").replace("+00:00", "Z")


def write(root, folder, name, minutes, *, day=0, model="model-a", parent=None,
          extra=()):
    """A pi session file with one message per entry of `minutes`
    (alternating user and assistant), in `root/folder/name.jsonl`."""
    d = root / folder
    d.mkdir(exist_ok=True)
    head = {"type": "session", "version": 3, "id": name, "cwd": "/work/x",
            "timestamp": iso(minutes[0], day)}
    if parent:
        head["parentSession"] = parent
    lines = [json.dumps(head)]
    for i, m in enumerate(minutes):
        msg = {"role": "user" if i % 2 == 0 else "assistant"}
        if i % 2:
            msg.update(model=model, usage={"input": 100, "cacheRead": 0,
                                           "cacheWrite": 0, "output": 5})
        lines.append(json.dumps({"type": "message", "timestamp": iso(m, day),
                                 "message": msg}))
    lines += list(extra)
    f = d / f"{name}.jsonl"
    f.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return f


def run(root, tmp_path, *args):
    out = tmp_path / "out.json"
    code = pc.main(["--sessions", str(root), "-o", str(out), *args])
    return code, json.loads(out.read_text())


def test_two_overlapping_sessions_give_the_mean_and_shares(tmp_path, capsys):
    # A in use 0-20 min, B 10-30: 1 session for 10 min, 2 for 10, 1 for 10
    write(tmp_path, "p1", "a", [0, 10, 20])
    write(tmp_path, "p2", "b", [10, 20, 30])
    code, r = run(tmp_path, tmp_path)
    c = r["concurrency"]
    assert code == 0
    assert c["mean"] == pytest.approx(4 / 3)
    assert c["share"]["1"] == pytest.approx(2 / 3)
    assert c["share"]["2"] == pytest.approx(1 / 3)
    assert (c["p50"], c["p90"], c["max"]) == (1, 2, 2)
    assert c["in_use_hours"] == pytest.approx(0.5)
    assert r["totals"]["root_sessions"] == 2
    day = r["days"]["list"]
    assert day == [{"date": "2026-03-10", "hours": 0.5, "peak": 2}]
    text = capsys.readouterr().out
    assert "mean 1.33" in text and "p90 2" in text


def test_a_gap_longer_than_idle_min_splits_a_session(tmp_path):
    # events 0, 10, 100, 110: two busy periods of 10 min each
    write(tmp_path, "p", "a", [0, 10, 100, 110])
    _, r = run(tmp_path, tmp_path)
    assert r["days"]["list"][0]["hours"] == pytest.approx(20 / 60, abs=1e-3)
    assert r["totals"]["busy_periods"] == 2
    _, r = run(tmp_path, tmp_path, "--idle-min", "120")
    assert r["days"]["list"][0]["hours"] == pytest.approx(110 / 60, abs=1e-3)
    assert r["totals"]["busy_periods"] == 1


def test_a_gap_of_exactly_idle_min_does_not_split(tmp_path):
    write(tmp_path, "p", "a", [0, 30])
    _, r = run(tmp_path, tmp_path)
    assert r["days"]["list"][0]["hours"] == pytest.approx(0.5, abs=1e-3)


def test_a_single_event_period_is_counted_but_lasts_zero(tmp_path):
    write(tmp_path, "p", "a", [0, 10, 200])      # [0,10] and one lone event
    _, r = run(tmp_path, tmp_path)
    assert r["totals"]["busy_periods"] == 2
    assert r["totals"]["busy_periods_of_one_event"] == 1
    assert r["concurrency"]["in_use_hours"] == pytest.approx(10 / 60)


def test_the_idle_min_lines_move_with_the_definition(tmp_path, capsys):
    write(tmp_path, "p1", "a", [0, 5, 10, 40, 45])
    write(tmp_path, "p2", "b", [40, 45, 50])
    _, r = run(tmp_path, tmp_path)
    by = {s["idle_min"]: s for s in r["idle_sensitivity"]}
    # idle 10: [0,10] x1, [40,45] x2, [45,50] x1 -> 25 / 20 min
    assert by[10]["mean"] == pytest.approx(1.25)
    # idle 30 and 60 join A's two pieces: 55 / 50 min
    assert by[30]["mean"] == pytest.approx(1.1)
    assert by[60]["mean"] == pytest.approx(1.1)
    out = capsys.readouterr().out
    assert " 10 min: mean 1.25" in out and " 60 min: mean 1.10" in out


def test_a_subagent_overlapping_its_parent_does_not_raise_concurrency(tmp_path):
    parent = write(tmp_path, "p", "main", [0, 10, 20])
    write(tmp_path, "p", "sub", [5, 15], parent=str(parent))
    _, r = run(tmp_path, tmp_path)
    t = r["totals"]
    assert (t["root_sessions"], t["subagent_files_merged"]) == (1, 1)
    assert r["concurrency"]["max"] == 1
    assert r["concurrency"]["in_use_hours"] == pytest.approx(20 / 60)


def test_a_subagent_extends_its_roots_time_and_follows_a_chain(tmp_path):
    root = write(tmp_path, "p", "main", [0, 10])
    mid = write(tmp_path, "p", "mid", [10, 20], parent=str(root))
    # a Windows-style absolute path to the parent still names the file
    win = "C:\\Users\\someone\\.pi\\agent\\sessions\\p\\" + mid.name
    write(tmp_path, "p", "leaf", [20, 40], parent=win)
    _, r = run(tmp_path, tmp_path)
    assert r["totals"]["root_sessions"] == 1
    assert r["totals"]["subagent_files_merged"] == 2
    assert r["concurrency"]["in_use_hours"] == pytest.approx(40 / 60)


def test_an_unresolvable_parent_is_counted_and_kept_as_its_own_session(tmp_path):
    write(tmp_path, "p", "a", [0, 10, 20])
    write(tmp_path, "p", "orphan", [5, 15],
          parent="/gone/2026-01-01T00-00-00Z_deleted.jsonl")
    _, r = run(tmp_path, tmp_path)
    t = r["totals"]
    assert t["parent_link_unresolved"] == 1
    assert t["root_sessions"] == 2 and t["subagent_files_merged"] == 0
    assert r["concurrency"]["max"] == 2


def test_a_parent_loop_is_counted_not_followed_forever(tmp_path):
    write(tmp_path, "p", "a", [0, 10], parent="/x/b.jsonl")
    write(tmp_path, "p", "b", [0, 10], parent="/x/a.jsonl")
    _, r = run(tmp_path, tmp_path)
    assert r["totals"]["parent_link_unresolved"] == 2
    assert r["totals"]["root_sessions"] == 2


def test_unreadable_unparseable_and_empty_files_are_counted(tmp_path, capsys):
    write(tmp_path, "p", "ok", [0, 10])
    (tmp_path / "p" / "dir.jsonl").mkdir()               # open() raises OSError
    (tmp_path / "p" / "junk.jsonl").write_text("not json\n{oops\n")
    (tmp_path / "p" / "header.jsonl").write_text(
        json.dumps({"type": "session", "id": "h"}) + "\n")
    write(tmp_path, "--tmp-run-", "scratch", [0, 10])
    write(tmp_path, "q", "torn", [0, 10], extra=['{"type": "mess'])
    code, r = run(tmp_path, tmp_path)
    t = r["totals"]
    assert code == 0
    assert t["session_files_found"] == 6
    assert t["session_files_read"] == 4
    assert t["files_skipped"] == {"unreadable": 1,
                                  "not parseable (no line is JSON)": 1,
                                  "no message with a timestamp": 1,
                                  "in a --tmp folder": 1}
    assert t["root_sessions"] == 2                      # ok and torn
    assert t["lines_that_are_not_json"] == 1
    assert t["files_with_lines_that_are_not_json"] == 1
    out = capsys.readouterr().out
    assert "1 unreadable" in out and "1 no message with a timestamp" in out


def test_a_message_with_a_bad_timestamp_is_counted(tmp_path):
    bad = json.dumps({"type": "message", "timestamp": "yesterday",
                      "message": {"role": "user"}})
    write(tmp_path, "p", "a", [0, 10], extra=[bad])
    _, r = run(tmp_path, tmp_path)
    assert r["totals"]["messages_without_a_readable_timestamp"] == 1
    assert r["concurrency"]["in_use_hours"] == pytest.approx(10 / 60)


def test_no_session_files_exits_naming_sessions(tmp_path):
    with pytest.raises(SystemExit) as e:
        pc.main(["--sessions", str(tmp_path / "missing"),
                 "-o", str(tmp_path / "o.json")])
    assert "--sessions" in str(e.value)


def test_model_keeps_sessions_with_90_percent_of_requests(tmp_path):
    # 10 assistant requests: 9 on the wanted model keeps, 8 drops
    def mixed(name, n_other):
        f = write(tmp_path, "p", name, list(range(0, 21)))
        rows = [json.loads(x) for x in f.read_text().splitlines()]
        k = 0
        for row in rows[1:]:
            if row["message"]["role"] == "assistant":
                row["message"]["model"] = ("other" if k < n_other
                                           else "glm-x")
                k += 1
        f.write_text("\n".join(json.dumps(x) for x in rows) + "\n")
    mixed("keep", 1)
    mixed("drop", 2)
    _, r = run(tmp_path, tmp_path, "--model", "GLM")
    assert r["totals"]["root_sessions"] == 1
    assert r["totals"]["files_skipped"] == {"left out by --model": 1}


def test_since_cuts_earlier_days(tmp_path):
    write(tmp_path, "p", "old", [0, 10], day=-5)
    write(tmp_path, "p", "new", [0, 10], day=0)
    _, r = run(tmp_path, tmp_path, "--since", "2026-03-08")
    t = r["totals"]
    assert t["root_sessions"] == 1
    assert t["files_skipped"] == {"before --since": 1}
    assert t["first_day"] == t["last_day"] == "2026-03-10"


def test_per_day_median_and_p90_and_the_list_holds_only_dates(tmp_path):
    for d, m in ((0, [0, 10]), (1, [0, 20]), (2, [0, 30])):
        write(tmp_path, "p", f"s{d}", m, day=d)
    write(tmp_path, "p", "t2", [10, 20], day=2)          # overlaps s2 on day 2
    _, r = run(tmp_path, tmp_path)
    d = r["days"]
    assert d["with_use"] == 3
    # hours: 10, 20, 30 min -> median 20 min
    assert d["hours_median"] == pytest.approx(20 / 60)
    assert d["peak_median"] == 1 and d["peak_p90"] == pytest.approx(1.8)
    assert [x["date"] for x in d["list"]] == ["2026-03-10", "2026-03-11",
                                              "2026-03-12"]
    assert set(r["totals"]) >= {"first_day", "last_day"}
    blob = json.dumps(r)
    assert "/work/x" not in blob and "model-a" not in blob


def test_a_run_with_no_time_in_use_exits_nonzero_and_says_so(tmp_path, capsys):
    write(tmp_path, "p", "a", [0])
    code, r = run(tmp_path, tmp_path)
    assert code == 1 and r["concurrency"] is None
    assert "no time with a session in use" in capsys.readouterr().out


def test_the_script_runs_alone_in_a_folder(tmp_path):
    only = tmp_path / "only"
    only.mkdir()
    shutil.copy(SCRIPT, only / "pi_concurrency.py")
    sessions = tmp_path / "sessions"
    sessions.mkdir()
    write(sessions, "p", "a", [0, 10, 20])
    r = subprocess.run([sys.executable, "-I", "pi_concurrency.py",
                        "--sessions", str(sessions)],
                       cwd=only, capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    assert (only / "pi-concurrency.json").exists()
    assert "root sessions: 1" in r.stdout


def test_shares_bin_four_apart_from_five_or_more(tmp_path):
    for i in range(4):
        write(tmp_path, "p", f"a{i}", [0, 10])
    for i in range(5):
        write(tmp_path, "p", f"b{i}", [100, 110])
    _, r = run(tmp_path, tmp_path)
    c = r["concurrency"]
    assert c["share"] == {"1": 0.0, "2": 0.0, "3": 0.0, "4": 0.5, "5+": 0.5}
    assert c["max"] == 5 and c["mean"] == pytest.approx(4.5)


def test_p50_is_the_level_that_reaches_half_the_time_exactly(tmp_path):
    # 10 min at 1 session, 10 min at 2: half the time is at or below 1
    write(tmp_path, "p", "a", [0, 10, 20])
    write(tmp_path, "p", "b", [10, 20])
    _, r = run(tmp_path, tmp_path)
    assert r["concurrency"]["p50"] == 1 and r["concurrency"]["p90"] == 2


def test_overlapping_busy_periods_of_one_root_are_not_counted_twice(tmp_path):
    parent = write(tmp_path, "p", "main", [0, 10, 20])
    for i in range(3):
        write(tmp_path, "p", f"sub{i}", [0, 20], parent=str(parent))
    _, r = run(tmp_path, tmp_path)
    assert r["concurrency"]["max"] == 1
    assert r["concurrency"]["in_use_hours"] == pytest.approx(20 / 60)

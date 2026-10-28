"""Milestone 4 baselines at live tables, operator-registered bot files, and benchmark runs."""

from __future__ import annotations

from pathlib import Path

import pytest

from arena.agents.baselines import BASELINES
from arena.agents.worker import artifact_hash
from tests.agents.conftest import BOTS, needs_sandbox
from tests.api.helpers import HUMAN, OPERATOR, auth, bot, harness

ROOT = Path(__file__).resolve().parents[2]


def test_catalog_lists_baselines_with_hashes_and_rulesets():
    with harness() as h:
        body = h.client.get("/catalog").json()
        by_id = {p["id"]: p for p in body["policies"]}
        assert body["operator"] is False
        for name in BASELINES:
            assert by_id[name]["source"] == "baseline"
            assert by_id[name]["artifact_hash"] == artifact_hash(BASELINES[name])
            assert by_id[name]["alias_of"] is None
        assert by_id["nlhe_heuristic"]["rulesets"] == ["practice_nlhe"]
        assert by_id["check_or_fold"]["alias_of"] == "check_fold"
        assert by_id["check_or_call"]["alias_of"] == "calling_station"


def test_unsupported_ruleset_is_rejected_before_the_table_exists():
    with harness() as h:
        body = {"config": {"ruleset_id": "uncertainty_plo4"}, "seats": [HUMAN, bot("nlhe_heuristic")]}
        r = h.client.post("/tables", json=body)
        assert r.status_code == 422
        assert r.json()["error"]["code"] == "unsupported_ruleset"
        assert r.json()["error"]["details"]["supported"] == ["practice_nlhe"]
        assert not h.manager.tables


@pytest.mark.parametrize("ruleset", ["practice_nlhe", "uncertainty_shortdeck", "uncertainty_plo4"])
def test_baseline_bots_finish_a_table(ruleset):
    field = ["check_fold", "calling_station", "legal_random"] + (["nlhe_heuristic"] if ruleset == "practice_nlhe" else [])
    with harness() as h:
        snap = h.setup([bot(p) for p in field], ruleset_id=ruleset, hand_cap=6, hands_per_level=2)
        assert snap["status"] == "complete"
        seats = snap["setup"]["seats"]
        assert [s["artifact_hash"] for s in seats] == [artifact_hash(BASELINES[p]) for p in field]
        assert {s["source"] for s in seats} == {"baseline"}
        assert not h.manager.tables[h.table_id].agents, "agents are closed when the table completes"


def test_registered_files_need_the_operator():
    with harness() as h:
        path = str(ROOT / "examples" / "template_bot.py")
        assert h.client.post("/bots", json={"path": path}).status_code == 401
        r = h.client.post("/bots", json={"path": path}, headers=auth("not-the-operator"))
        assert r.status_code == 403 and r.json()["error"]["code"] == "operator_required"
        assert h.client.get("/bots").status_code == 401
        assert h.client.get("/operator", headers=auth(OPERATOR)).json() == {"operator": True}
        assert h.client.get("/runs").status_code == 401
        assert h.client.post("/runs", json={"cells": [], "bots": []}).status_code == 401
        assert h.client.get("/runs/r_nope").status_code == 401


def test_relative_and_missing_paths_are_rejected():
    with harness() as h:
        for path, code in (("examples/template_bot.py", "invalid_path"), ("/nonexistent/bot.py", "not_found")):
            r = h.client.post("/bots", json={"path": path}, headers=auth(OPERATOR))
            assert r.status_code == 422 and r.json()["error"]["code"] == code


@needs_sandbox
def test_registered_bot_is_sandboxed_operator_only_and_plays():
    with harness() as h:
        path = BOTS / "plo_only_bot.py"
        r = h.client.post("/bots", json={"path": str(path)}, headers=auth(OPERATOR))
        assert r.status_code == 201, r.text
        spec = r.json()["bot"]
        assert spec["id"] == "file_" + artifact_hash(path)[:12]
        assert spec["source"] == "registered" and spec["rulesets"] == ["uncertainty_plo4"]
        again = h.client.post("/bots", json={"path": str(path)}, headers=auth(OPERATOR)).json()["bot"]
        assert again["id"] == spec["id"]

        public = {p["id"] for p in h.client.get("/catalog").json()["policies"]}
        assert spec["id"] not in public
        as_operator = h.client.get("/catalog", headers=auth(OPERATOR)).json()
        assert spec["id"] in {p["id"] for p in as_operator["policies"]} and as_operator["operator"]
        assert "original_path" not in next(p for p in h.client.get("/catalog").json()["policies"])

        seats = [bot(spec["id"]), bot("legal_random")]
        anon = h.client.post("/tables", json={"config": {"ruleset_id": "uncertainty_plo4"}, "seats": seats})
        assert anon.status_code == 422 and anon.json()["error"]["code"] == "invalid_seat"
        wrong = h.client.post("/tables", json={"config": {}, "seats": seats}, headers=auth(OPERATOR))
        assert wrong.json()["error"]["code"] == "unsupported_ruleset"

        snap = h.setup(seats, operator=True, ruleset_id="uncertainty_plo4", hand_cap=3)
        assert snap["status"] == "complete"
        assert snap["setup"]["seats"][0]["source"] == "registered"
        assert snap["setup"]["seats"][0]["artifact_hash"] == spec["artifact_hash"]
        assert "original_path" not in snap["setup"]["seats"][0]


def _run(h, **body):
    r = h.client.post("/runs", json=body, headers=auth(OPERATOR))
    return r


def test_run_rejections():
    with harness() as h:
        cases = [
            ({"cells": [{"ruleset_id": "uncertainty_plo4", "players": 2}], "bots": ["nlhe_heuristic"]}, "unsupported_ruleset"),
            ({"cells": [{"ruleset_id": "practice_nlhe", "players": 2}], "bots": ["nope"]}, "unknown_bot"),
            ({"cells": [{"ruleset_id": "practice_nlhe", "players": 2}], "bots": ["legal_random"],
              "config": {"initial_player_count": 3}}, "invalid_config"),
            ({"cells": [{"ruleset_id": "practice_nlhe", "players": 2}], "bots": ["legal_random"],
              "config": {"hand_cap": 0}}, "invalid_config"),
            ({"cells": [{"ruleset_id": "practice_nlhe", "players": 2}] * 2, "bots": ["legal_random"]}, "duplicate_cell"),
        ]
        for body, code in cases:
            r = _run(h, **body)
            assert r.status_code == 422, (body, r.text)
            assert r.json()["error"]["code"] == code
        assert _run(h, cells=[{"ruleset_id": "practice_nlhe", "players": 9}], bots=["legal_random"]).status_code == 422


def test_run_launches_the_benchmark_runner_in_the_background(tmp_path):
    with harness(runs_root=tmp_path) as h:
        r = _run(h, cells=[{"ruleset_id": "practice_nlhe", "players": 2},
                           {"ruleset_id": "uncertainty_plo4", "players": 3}],
                 bots=["calling_station", "legal_random"], blocks=2, isolation="in_process", seed="runs-test",
                 config={"hand_cap": 4, "hands_per_level": 2})
        assert r.status_code == 202, r.text
        run_id = r.json()["run_id"]
        assert r.json()["progress"]["total"] == 2 * 2 + 2 * 3
        done = h.runs.wait(run_id, 120)
        assert done["status"] == "complete", done
        got = h.client.get(f"/runs/{run_id}", headers=auth(OPERATOR)).json()
        assert got["progress"] == {"done": 10, "total": 10}
        assert "seed" not in got["request"]
        nlhe, plo = got["cells"]
        assert (nlhe["cell_id"], plo["cell_id"]) == ("practice_nlhe-2p", "uncertainty_plo4-3p")
        assert [e["name"] for e in plo["entrants"]] == ["calling_station#1", "legal_random", "calling_station#2"]
        m = plo["manifest"]
        assert m["isolation"] == "in_process" and m["competitive"] is False
        assert m["summary"]["tables_completed"] == 6 and m["summary"]["tables_failed"] == []
        assert m["config"]["ruleset_id"] == "uncertainty_plo4" and len(m["config_hash"]) == 64
        assert {e["artifact_sha256"] for e in m["entrants"]} == {artifact_hash(BASELINES["calling_station"]),
                                                                 artifact_hash(BASELINES["legal_random"])}
        for e in m["summary"]["entrants"]:
            assert e["tables"] == 6 and e["mean_points"] is not None and "timing" in e and "fallbacks" in e
        assert (tmp_path / run_id / "uncertainty_plo4-3p" / "manifest.json").is_file()
        listed = h.client.get("/runs", headers=auth(OPERATOR)).json()["runs"]
        assert listed[0]["run_id"] == run_id and listed[0]["scenarios"] == ["practice_nlhe · 2 seats",
                                                                           "uncertainty_plo4 · 3 seats"]
        assert h.client.get("/runs/r_missing", headers=auth(OPERATOR)).status_code == 404


def test_in_process_runs_refuse_registered_files(tmp_path):
    with harness(runs_root=tmp_path) as h:
        h.manager.bots.registered["file_x"] = h.manager.bots.baselines["legal_random"].__class__(
            **{**h.manager.bots.baselines["legal_random"].__dict__, "id": "file_x", "source": "registered"})
        r = _run(h, cells=[{"ruleset_id": "practice_nlhe", "players": 2}], bots=["file_x"], isolation="in_process")
        assert r.status_code == 422 and r.json()["error"]["code"] == "isolation_required"

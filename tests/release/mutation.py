"""Seeded-defect mutation checks (spec 12.4).

Each mutant is a small source patch applied to a *test-only copy* of the code
(``arena/``, ``tests/``, ``pyproject.toml`` copied into a temporary directory;
the working tree is never modified). Every replacement must match exactly once,
so a refactor that moves the target makes the shard fail loudly instead of
silently testing nothing. For each mutant the shard:

1. checks that the copy imports ``arena`` from the temporary directory;
2. runs the mutant's named tests on the unmodified copy: they must pass;
3. applies the patch and reruns them: at least one named test must *fail*
   (an assertion or a raised engine discrepancy), not merely error during
   collection.

A mutant that survives fails the gate.
"""

from __future__ import annotations

import hashlib
import os
import shutil
import subprocess
import sys
import tempfile
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class Mutant:
    id: str
    defect: str
    """The spec 12.4 wording."""
    file: str
    patches: tuple[tuple[str, str], ...]
    tests: tuple[str, ...]
    """Named tests (pytest node ids) that must catch the defect."""
    note: str = ""


ADAPTER = "arena/rules/pokerkit_adapter.py"

MUTANTS: list[Mutant] = [
    Mutant(
        "hu_action_order", "swap HU action order", ADAPTER,
        (("        return (spec.big_blind_seat, spec.button)\n",
          "        return (spec.button, spec.big_blind_seat)\n"),),
        ("tests/golden/test_core_fixtures.py::test_core_fixture[G01]",),
        "heads-up engine order puts the button/SB in the BB's engine slot, so the BB acts first preflop",
    ),
    Mutant(
        "raise_debits_target", "debit raise target instead of delta", ADAPTER,
        (("                _strict(self.pk.complete_bet_or_raise_to, target)\n"
          "                if increment >= self._full_increment:\n",
          "                _strict(self.pk.complete_bet_or_raise_to, target)\n"
          "                self.pk.stacks[i] -= bet\n"
          "                if increment >= self._full_increment:\n"),),
        ("tests/property/test_chip_accounting.py::test_raise_to_debits_only_the_delta",
         "tests/golden/test_core_fixtures.py::test_core_fixture[G14]"),
        "a raise to T by a seat that already has B in front of it debits T chips instead of T - B",
    ),
    Mutant(
        "short_allin_reopens", "reopen after every short all-in", ADAPTER,
        (("        return acted is None or level - acted >= self._full_increment\n",
          "        return acted is None or level > acted\n"),),
        ("tests/golden/test_core_fixtures.py::test_core_fixture[G09b]",
         "tests/golden/test_core_fixtures.py::test_core_fixture[G21a]"),
        "any raise since a seat last acted, however short, gives it raise rights again",
    ),
    Mutant(
        "drop_folded_contributions", "drop folded contributions", ADAPTER,
        (("            contributors = [s for s in self.seats if contributions[s] >= level]\n",
          "            contributors = [s for s in self.seats if contributions[s] >= level and s not in self.folded]\n"),),
        ("tests/golden/test_core_fixtures.py::test_core_fixture[G12]",),
        "pot layers count only live seats, so chips put in by folded players vanish from the pots",
    ),
    Mutant(
        "excess_as_side_pot", "award unmatched excess as a side pot", ADAPTER,
        (("            if ev.type == \"BetsCollected\":\n"
          "                for seat, amount in ev.data[\"bets\"].items():\n"
          "                    out[seat] += amount\n",
          "            if ev.type == \"BetsCollected\":\n"
          "                for seat, amount in ev.data[\"bets\"].items():\n"
          "                    out[seat] += amount\n"
          "            elif ev.type == \"ExcessReturned\":\n"
          "                out[ev.data[\"seat\"]] += ev.data[\"amount\"]\n"),
         ("            if len(contributors) < 2:\n"
          "                raise EngineDiscrepancy(\"unmatched excess left in the pot after collection\")\n",
          "            if len(contributors) < 2:\n"
          "                pass\n")),
        ("tests/golden/test_core_fixtures.py::test_core_fixture[G13]",),
        "the uncalled part of a bet is kept as a one-player pot layer and awarded instead of only being returned",
    ),
    Mutant(
        "icm_denominator", "change an ICM denominator", "arena/tournament/scoring.py",
        (("total = sum(stacks[i] for i in members)", "total = sum(stacks)"),),
        ("tests/golden/test_tournament_fixtures.py::test_icm_fixture[T04]",),
        "finish probabilities divide by every initial stack instead of the stacks still in the subset",
    ),
    Mutant(
        "expose_hidden_card", "expose a hidden card", "arena/table/projections.py",
        (("            shown = hand.revealed.get(seat)\n",
          "            shown = hand.revealed.get(seat) or tuple(hand.hole_cards.get(seat, ()))\n"),),
        ("tests/property/test_information_sets.py::test_views_identical_across_hidden_card_worlds",
         "tests/api/test_leaks.py::test_no_unauthorized_cards_in_any_body_frame_or_error"),
        "every player's hole cards appear as shown cards in every view",
    ),
    Mutant(
        "duplicate_applied_twice", "apply a duplicate action twice", "arena/api/manager.py",
        (("        prior = live.receipts.get(envelope.action_id)\n        if prior is not None:\n",
          "        prior = None\n        if prior is not None:\n"),),
        ("tests/api/test_tables_api.py::test_action_id_idempotency",
         "tests/persistence/test_faults.py::test_duplicate_identical_actions_racing_commit_once[memory]"),
        "a retried action_id skips the receipt check and is submitted to the table again",
    ),
]
MUTANT_BY_ID = {m.id: m for m in MUTANTS}


def _copy_tree(dest: Path) -> None:
    ignore = shutil.ignore_patterns("__pycache__", "*.pyc", "node_modules", ".pytest_cache")
    for name in ("arena", "tests", "examples"):
        if (ROOT / name).exists():
            shutil.copytree(ROOT / name, dest / name, ignore=ignore)
    for name in ("pyproject.toml",):
        shutil.copy2(ROOT / name, dest / name)


def apply(mutant: Mutant, dest: Path) -> None:
    path = dest / mutant.file
    text = path.read_text()
    for old, new in mutant.patches:
        hits = text.count(old)
        if hits != 1:
            raise RuntimeError(f"mutant {mutant.id}: patch target matches {hits} times in {mutant.file} "
                               "(expected exactly once); update the mutant to the current code")
        text = text.replace(old, new)
    path.write_text(text)


def _env(dest: Path) -> dict[str, str]:
    env = {k: v for k, v in os.environ.items() if not k.startswith("PYTEST")}
    env["PYTHONPATH"] = str(dest)
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    env.pop("ARENA_API_TEST_STORE", None)
    return env


def _pytest(dest: Path, tests: list[str], junit: Path) -> tuple[int, list[dict[str, str]], str]:
    proc = subprocess.run([sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider", "-o",
                           "junit_family=xunit1", f"--junitxml={junit}", *tests],
                          cwd=dest, env=_env(dest), capture_output=True, text=True, timeout=1800)
    cases = []
    if junit.exists():
        for tc in ET.parse(junit).getroot().iter("testcase"):
            outcome = "passed"
            message = ""
            for child in tc:
                if child.tag in ("failure", "error", "skipped"):
                    outcome = {"failure": "failed", "error": "error", "skipped": "skipped"}[child.tag]
                    message = (child.get("message") or "")[:300]
            cases.append({"id": f"{tc.get('classname')}::{tc.get('name')}", "outcome": outcome, "message": message})
    return proc.returncode, cases, (proc.stdout + proc.stderr)[-2000:]


def check_mutant(mutant: Mutant, tests: list[str] | None = None) -> dict[str, Any]:
    tests = list(tests or mutant.tests)
    with tempfile.TemporaryDirectory(prefix=f"arena-mutant-{mutant.id}-") as tmp:
        dest = Path(tmp)
        _copy_tree(dest)
        where = subprocess.run([sys.executable, "-c", "import arena; print(arena.__file__)"], cwd=dest,
                               env=_env(dest), capture_output=True, text=True).stdout.strip()
        if not where.startswith(str(dest)):
            raise RuntimeError(f"copy imports arena from {where}, not the test-only copy {dest}")
        clean_code, clean, clean_out = _pytest(dest, tests, dest / "clean.xml")
        apply(mutant, dest)
        mut_code, mutated, mut_out = _pytest(dest, tests, dest / "mutant.xml")
    caught_by = [c["id"] for c in mutated if c["outcome"] == "failed"]
    return {
        "id": mutant.id, "defect": mutant.defect, "file": mutant.file, "note": mutant.note, "tests": tests,
        "clean_exit": clean_code, "clean_outcomes": {c["id"]: c["outcome"] for c in clean},
        "mutant_exit": mut_code, "mutant_outcomes": {c["id"]: c["outcome"] for c in mutated},
        "caught_by": caught_by,
        "first_failure": next((c["message"] for c in mutated if c["outcome"] == "failed"), ""),
        "clean_output_tail": clean_out if clean_code else "",
        "mutant_output_tail": mut_out if not caught_by else "",
    }


def run_shard(params: dict[str, Any]) -> dict[str, Any]:
    mutant = MUTANT_BY_ID[params["mutant"]]
    out = check_mutant(mutant)
    failures = []
    clean_ok = out["clean_exit"] == 0 and out["clean_outcomes"] and all(
        v == "passed" for v in out["clean_outcomes"].values())
    if not clean_ok:
        failures.append(f"named tests do not pass on the unmodified copy: {out['clean_outcomes']} "
                        f"{out['clean_output_tail'][-800:]}")
    if not out["caught_by"]:
        failures.append(f"mutant survived: named tests {out['mutant_outcomes']} {out['mutant_output_tail'][-800:]}")
    return {
        "cases": 1,
        "failed_cases": len(failures),
        "counts": {"mutants": 1, "mutants_killed": int(bool(out["caught_by"])), "named_tests": len(out["tests"])},
        "exclusions": {},
        "failures": [{"case": {"gate": "mutation", "mutant": mutant.id}, "error": f, "traceback": "",
                      "repro": f"uv run python -c \"from tests.release.mutation import *; "
                               f"print(check_mutant(MUTANT_BY_ID['{mutant.id}']))\""} for f in failures],
        "result_hash": hashlib.sha256(f"{mutant.id}|{sorted(out['caught_by'])}|{clean_ok}".encode()).hexdigest(),
        "mutants": {k: out[k] for k in ("id", "defect", "file", "note", "tests", "caught_by", "first_failure")},
    }


def shards() -> list[dict[str, Any]]:
    return [{"gate": "mutation", "mutant": m.id} for m in MUTANTS]


def finalize(gate: str, results: list[dict[str, Any]], all_results: dict) -> list[dict[str, Any]]:
    killed = {r["mutants"]["id"]: r["mutants"]["caught_by"] for r in results}
    return [{"check": "all eight spec 12.4 seeded defects caught by a named test",
             "passed": len(killed) == len(MUTANTS) and all(killed.values()),
             "detail": {k: (v[0] if v else "SURVIVED") for k, v in killed.items()}}]


def explore(mutant_id: str, paths: list[str]) -> list[str]:
    """Development helper: which tests in ``paths`` fail under a mutant."""
    out = check_mutant(MUTANT_BY_ID[mutant_id], paths)
    return out["caught_by"]

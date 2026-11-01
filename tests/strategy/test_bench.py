import json

from strategy.solve import bench
from tests.strategy.helpers import kuhn_tree


def test_quick_benchmark_writes_a_comparable_report(tmp_path, capsys):
    assert bench.main(["--quick", "--games", "kuhn", "--out", str(tmp_path)]) == 0
    summary = (tmp_path / "summary.md").read_text()
    for method in bench.METHODS:
        assert f"| {method} |" in summary
    runs = json.loads((tmp_path / "kuhn.json").read_text())
    assert [r["method"] for r in runs] == list(bench.METHODS)
    budget = bench.GAMES["kuhn"].quick_budget
    for r in runs:
        last = r["checkpoints"][-1]
        assert last["nodes_touched"] >= budget
        nodes = [c["nodes_touched"] for c in r["checkpoints"]]
        assert nodes == sorted(nodes)
    assert "Nodes to 0.01" in capsys.readouterr().out


def test_full_traversal_counts_one_tree_walk_per_player_pass():
    tree = kuhn_tree()
    run = bench.run_method(tree, "kuhn", "cfr", budget=10 * 2 * tree.num_nodes)
    last = run.checkpoints[-1]
    assert last.nodes_touched == last.iterations * 2 * tree.num_nodes

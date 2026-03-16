import json

from scripts.analyze_runs import find_runs, load_data_from_logdir


def test_load_data_from_json_logdir(tmp_path):
    run_dir = tmp_path / "runs" / "demo_run"
    run_dir.mkdir(parents=True)
    with open(run_dir / "demo_run_data.json", "w", encoding="utf-8") as f:
        json.dump({"charts/success_rate": [[1, 0.25], [2, 0.5]]}, f)

    data = load_data_from_logdir(str(run_dir), tags=["charts/success_rate"])

    assert "charts/success_rate" in data
    assert data["charts/success_rate"]["step"].tolist() == [1, 2]
    assert data["charts/success_rate"]["value"].tolist() == [0.25, 0.5]


def test_load_data_from_json_prefers_candidate_with_requested_tags(tmp_path):
    run_root = tmp_path / "runs" / "demo_run"
    bad_dir = run_root / "brain_trends"
    good_dir = run_root / "episode_1"
    bad_dir.mkdir(parents=True)
    good_dir.mkdir(parents=True)

    with open(bad_dir / "brain_data.json", "w", encoding="utf-8") as f:
        json.dump({"brain/episode_reward": [[1, 1.0]]}, f)
    with open(good_dir / "inner_data.json", "w", encoding="utf-8") as f:
        json.dump({"charts/success_rate": [[3, 0.75]]}, f)

    data = load_data_from_logdir(str(run_root), tags=["charts/success_rate"])

    assert data["charts/success_rate"]["step"].tolist() == [3]
    assert data["charts/success_rate"]["value"].tolist() == [0.75]


def test_find_runs_detects_json_logs(tmp_path, monkeypatch):
    run_dir = tmp_path / "runs" / "demo_run"
    run_dir.mkdir(parents=True)
    (run_dir / "demo_run_data.json").write_text("{}", encoding="utf-8")

    monkeypatch.chdir(tmp_path)
    runs = find_runs()

    assert ("demo_run", str((tmp_path / "runs" / "demo_run").relative_to(tmp_path))) in runs

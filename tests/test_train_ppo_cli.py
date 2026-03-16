import sys

import scripts.train_ppo as train_ppo_script


def test_train_ppo_forwards_replay_flags(monkeypatch):
    captured = {}

    def fake_train_ppo(**kwargs):
        captured.update(kwargs)

    monkeypatch.setattr(train_ppo_script, "train_ppo", fake_train_ppo)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "train_ppo.py",
            "--replay_ratio",
            "0.25",
            "--replay_prioritization",
            "0.75",
            "--run_name",
            "cli_test",
        ],
    )

    train_ppo_script.main()

    assert captured["replay_ratio"] == 0.25
    assert captured["replay_prioritization"] == 0.75

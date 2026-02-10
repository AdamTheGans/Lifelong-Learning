"""Analyze training metrics from JSONL log files."""
import json
import sys

def analyze(fname, label):
    lines = open(fname).readlines()
    print(f"\n{'='*60}")
    print(f"  {label}  ({len(lines)} log entries)")
    print(f"{'='*60}")
    
    # Separate episode metrics and training metrics
    ep_entries = []
    train_entries = []
    
    for line in lines:
        d = json.loads(line)
        s = d.get("step", 0)
        
        sc = d.get("episode/score")
        if sc is not None:
            ep_entries.append({
                "step": s,
                "score": sc,
                "length": d.get("episode/length", 0),
            })
        
        g = d.get("epstats/log/reached_good_goal/max")
        if g is not None:
            train_entries.append({
                "step": s,
                "good": g,
                "bad": d.get("epstats/log/reached_bad_goal/max", 0),
                "tout": d.get("epstats/log/timed_out/max", 0),
                "img": d.get("train/loss/image", None),
                "rew": d.get("train/loss/rew", None),
                "dyn": d.get("train/loss/dyn", None),
                "grad": d.get("train/opt/grad_norm", None),
                "regime": d.get("epstats/log/regime_id/avg", None),
            })
    
    print(f"\n  Episode entries: {len(ep_entries)}")
    print(f"  Training entries with epstats: {len(train_entries)}")
    
    # Show episode score summary
    if ep_entries:
        print(f"\n  --- Episode Scores ---")
        n = len(ep_entries)
        for idx in [0, n//5, 2*n//5, 3*n//5, 4*n//5, n-1]:
            e = ep_entries[idx]
            print(f"    step={e['step']:>6}  score={e['score']:>7.2f}  length={e['length']:>5.0f}")
    
    # Show epstats
    if train_entries:
        print(f"\n  --- Epstats (goal rates) ---")
        for e in train_entries:
            img_s = f"{e['img']:.0f}" if e['img'] is not None else "."
            rew_s = f"{e['rew']:.2f}" if e['rew'] is not None else "."
            dyn_s = f"{e['dyn']:.1f}" if e['dyn'] is not None else "."
            grad_s = f"{e['grad']:.0f}" if e['grad'] is not None else "."
            reg_s = f"{e['regime']:.1f}" if e['regime'] is not None else "."
            print(f"    step={e['step']:>6}  good={e['good']:.2f}  bad={e['bad']:.2f}  tout={e['tout']:.2f}  img={img_s:>5}  rew={rew_s:>6}  dyn={dyn_s:>5}  grad={grad_s:>6}  reg={reg_s}")
    
    # Compute overall averages for first vs last quarter
    if ep_entries:
        q1 = ep_entries[:len(ep_entries)//4]
        q4 = ep_entries[3*len(ep_entries)//4:]
        avg_q1 = sum(e['score'] for e in q1) / len(q1)
        avg_q4 = sum(e['score'] for e in q4) / len(q4)
        avg_len_q1 = sum(e['length'] for e in q1) / len(q1)
        avg_len_q4 = sum(e['length'] for e in q4) / len(q4)
        print(f"\n  --- Score trend ---")
        print(f"    First quarter avg score  = {avg_q1:.2f}  avg_len = {avg_len_q1:.0f}")
        print(f"    Last  quarter avg score  = {avg_q4:.2f}  avg_len = {avg_len_q4:.0f}")

analyze("baseline_50k/metrics.jsonl", "BASELINE (stationary, 50k steps)")
analyze("switch_15k_50k/metrics.jsonl", "SWITCH (15k regime, 50k steps)")

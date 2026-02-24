#!/usr/bin/env python3
"""
Launcher script for the Multi-Task Oracle test.
This ensures the agent receives frequent interleaving and perfect routing.
"""
import sys
import subprocess

def main():
    # Base command for the oracle test
    cmd = [
        sys.executable,
        "scripts/train_ppo.py",
        "--oracle",
        "--steps_per_regime", "1000",
        "--run_name", "oracle_baseline"
    ]
    
    # Allow the user to override or add arguments (e.g., --total_timesteps)
    cmd.extend(sys.argv[1:])
    
    print("=====================================================")
    print("Launching Oracle Baseline Test")
    print("=====================================================")
    print(f"Command: {' '.join(cmd)}")
    
    try:
        subprocess.run(cmd, check=True)
    except KeyboardInterrupt:
        print("\nTraining interrupted by user.")
    except subprocess.CalledProcessError as e:
        print(f"\nTraining script failed with exit code: {e.returncode}")
        sys.exit(e.returncode)

if __name__ == "__main__":
    main()

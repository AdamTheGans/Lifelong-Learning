import json
import os
import sys

def main():
    folder = "evals/eval_hyperparams_3regime_20260305-021618"
    file_path = os.path.join(folder, "eval_hyperparams_ep1_env0_data.json")
    
    with open(file_path, "r") as f:
        data = json.load(f)
    
    with open("_inspect_out.txt", "w") as out:
        out.write(f"Type: {type(data)}\n")
        if isinstance(data, list):
            out.write(f"Length: {len(data)}\n")
            if len(data) > 0:
                out.write("First item:\n")
                out.write(json.dumps(data[0], indent=2) + "\n")
                keys = set()
                for item in data[:100]:
                    keys.update(item.keys())
                out.write(f"Keys in first 100 items: {keys}\n")
        elif isinstance(data, dict):
            out.write(f"Keys: {list(data.keys())}\n")
            for k, v in data.items():
                out.write(f"Key {k} type: {type(v)}\n")
                if isinstance(v, list) and len(v) > 0:
                    out.write(f"Key {k} first item type: {type(v[0])}\n")

if __name__ == "__main__":
    main()

import argparse
import json
import os

def visualize_log(input_path: str, output_path: str = None):
    with open(input_path) as f:
        log = f.readlines()
        log = [l[:-1] + ",\n" for l in log]
        log[-1] = log[-1][:-2]
        log = json.loads("[" + "".join(log) + "]")
    
    lines = []
    lines_actions = []
    for entry in log:
        lines.append(str(entry["step"]) + ": ")
        lines.append(entry["action"] + "\n")
        lines_actions.append(str(entry["step"]) + ": ")
        lines_actions.append(entry["action"] + "\n")
        lines.append(entry["observation"]["text"] + "\n")
        lines.append(entry["response"] + "\n")
        lines.append("==" * 40 + "\n")
    
    if output_path is None:
        base = os.path.splitext(os.path.basename(input_path))[0]
        output_path = f"{base}_visualized.txt"
    
    with open(output_path, "w") as f:
        f.writelines(lines)
    
    actions_path = f"{os.path.splitext(output_path)[0]}_actions.txt"
    with open(actions_path, "w") as f:
        f.writelines(lines_actions)
    
    print(f"Written full log to: {output_path}")
    print(f"Written actions only to: {actions_path}")


def main():
    parser = argparse.ArgumentParser(description="Visualize agent log files")
    parser.add_argument("input", help="Path to agent_log.jsonl file")
    parser.add_argument("-o", "--output", help="Output file path (default: <input>_visualized.txt)")
    args = parser.parse_args()
    
    visualize_log(args.input, args.output)


if __name__ == "__main__":
    main()

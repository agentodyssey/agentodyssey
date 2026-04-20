import os
import json
import random
import tempfile
import importlib

def atomic_write(path: str, data: str):
    dir_name = os.path.dirname(path)
    with tempfile.NamedTemporaryFile("w", delete=False, dir=dir_name, encoding="utf-8") as tmp_file:
        tmp_file.write(data)
        temp_name = tmp_file.name
    os.replace(temp_name, path)

def load_config(config_path, from_step=None):
    is_jsonl = config_path.lower().endswith(".jsonl")
    if not is_jsonl:
        with open(config_path, "r", encoding="utf-8") as f:
            return json.load(f)
    # JSONL handling
    with open(config_path, "r", encoding="utf-8-sig") as f:
        lines = [line.strip() for line in f if line.strip()]
    if not lines:
        return None
    if from_step is None:
        return json.loads(lines[-1])
    # interpret from_step as the actual env step number
    for line in reversed(lines):  # reverse for speed (latest first)
        obj = json.loads(line)
        if obj.get("step", None) == from_step:
            return obj
    raise ValueError(f"Step {from_step} not found in {config_path}")

def get_def_id(obj_id):
    if obj_id.split("_")[-1].isdigit():
        return "_".join(obj_id.split("_")[:-1])
    return obj_id

def dynamic_load_game_class(game_name: str, module_name: str, class_name: str):
    if game_name in (None, "", "base"):
        module_name = f"games.base.{module_name}"
    else:
        module_name = f"games.generated.{game_name}.{module_name}"

    module = importlib.import_module(module_name)
    try:
        return getattr(module, class_name)
    except AttributeError as e:
        raise ValueError(f"Class {class_name} not found in {module_name}") from e

def build_choices_with_answer_idx(answer, distractor_pool: set, max_choices: int = 10) -> tuple[list[str], int]:
    answer_str = str(answer)
    pool = {str(x) for x in distractor_pool if str(x)}
    pool.discard(answer_str)
    distractors = random.sample(list(pool), min(max(0, max_choices - 1), len(pool)))
    choices = distractors + [answer_str]
    random.shuffle(choices)
    answer_idx = choices.index(answer_str)
    return choices, answer_idx

def get_hardware_info():
    import platform, subprocess
    import psutil

    def safe_import(name):
        try:
            return __import__(name)
        except Exception:
            return None
    
    cpuinfo = safe_import("cpuinfo")
    GPUtil = safe_import("GPUtil")
    pynvml = safe_import("pynvml")
    # --- CPU ---
    cpu_name = None
    try:
        if cpuinfo:
            cpu_name = cpuinfo.get_cpu_info().get("brand_raw")
        if not cpu_name:
            cpu_name = platform.processor() or "Unknown CPU"
    except Exception:
        cpu_name = "Unknown CPU"
    # --- Memory ---
    try:
        mem = psutil.virtual_memory().total / (1024 ** 3)
        mem_size = f"{mem:.1f} GB"
    except Exception:
        mem_size = "Unknown"
    # --- GPU(s) ---
    gpu_names = []
    try:
        if GPUtil:
            gpus = GPUtil.getGPUs()
            gpu_names = [g.name for g in gpus]
        elif pynvml:
            pynvml.nvmlInit()
            for i in range(pynvml.nvmlDeviceGetCount()):
                h = pynvml.nvmlDeviceGetHandleByIndex(i)
                gpu_names.append(pynvml.nvmlDeviceGetName(h).decode())
            pynvml.nvmlShutdown()
        else:
            out = subprocess.getoutput("lspci | grep -i vga")
            if out:
                gpu_names = [line.split(":")[-1].strip() for line in out.splitlines()]
    except Exception:
        pass
    return {"cpu": cpu_name, "memory": mem_size, "gpus": gpu_names or ["No GPU detected"]}

def convert_json_to_jsonl(seed_json_path, jsonl_path):
    import json
    with open(seed_json_path, "r") as f:
        obj = json.load(f)
    with open(jsonl_path, "w") as f:
        f.write(json.dumps(obj) + "\n")

if __name__ == "__main__":
    hardware = get_hardware_info()
    print("CPU:", hardware["cpu"])
    print("Memory:", hardware["memory"])
    print("GPUs:", ", ".join(hardware["gpus"]))
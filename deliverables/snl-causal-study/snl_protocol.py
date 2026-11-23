"""Keep formal SNL label acquisition and scoring tied to the same frozen run."""
from contextlib import contextmanager

from snl_inputs import BLIND, DEFAULT_OUTPUT, ROOT, VIDEO, read_json, sha_file, write_json

GATE = 0.80


def passed(result):
    return (result["recall"] or 0) >= GATE and (result["precision"] or 0) >= GATE


def causal_prefix_length(config):
    return 96 if config.get("input_profile") == "stacked-long" else 36


@contextmanager
def exclusive_run(key):
    DEFAULT_OUTPUT.mkdir(parents=True, exist_ok=True)
    lock = DEFAULT_OUTPUT / f"blind-{key}.lock"
    try:
        lock.touch(exist_ok=False)
    except FileExistsError as error:
        raise ValueError(f"Another operation owns the {key} blind split") from error
    try:
        yield
    finally:
        lock.unlink()


def selection(output, key):
    choice = read_json(output / "development-status.json")
    selected = choice["best"]
    if choice["status"] != "ready" or not all(passed(selected[name]) for name in ("validation", "development")):
        raise ValueError("Both development splits must reach the stipulated gate before opening a blind split")
    config = selected["config"]
    prefix = output / "causality-check.json"
    if not prefix.is_file() or read_json(prefix) != causal_receipt(config):
        raise ValueError("The selected version must pass the causal prefix check before freezing")
    if key == "reserve":
        originals = list(DEFAULT_OUTPUT.glob("**/blind-first.json"))
        if len(originals) != 1 or read_json(originals[0])["passed"]:
            raise ValueError("Reserve scoring follows an unsuccessful first blind evaluation")
        if read_json(originals[0])["config"] == config:
            raise ValueError("Reserve scoring requires a different frozen configuration")
    checkpoint = output / config["path"] if "path" in config else None
    if checkpoint is not None and sha_file(checkpoint) != config["sha256"]:
        raise ValueError("The selected checkpoint digest differs")
    return {"output": str(output), "selected": selected,
            "label_blocks": {name: list(ids) for name, ids in BLIND[key].items()},
            "video_sha256": sha_file(VIDEO), "inference_sha256": inference_hashes()}


def inference_hashes():
    scripts = ("experiment.py", "snl_inputs.py", "snl_detector.py", "snl_center.py",
               "snl_background.py", "snl_dense.py", "snl_protocol.py", "snl_yolo.py")
    hashes = {name: sha_file(ROOT / "deliverables/snl-causal-study" / name) for name in scripts}
    hashes["videosar/snl_causal.py"] = sha_file(ROOT / "videosar/snl_causal.py")
    return hashes


def causal_receipt(config):
    return {"passed": True, "prefix_frames": causal_prefix_length(config), "seam_check": "passed",
            "config": config, "inference_sha256": inference_hashes()}


def freeze_selection(output, key, *, create=False):
    if (output / f"blind-{key}.json").exists() or any(DEFAULT_OUTPUT.glob(f"**/blind-{key}.json")):
        raise ValueError("This blind split already has a formal outcome")
    current = selection(output, key)
    receipt = DEFAULT_OUTPUT / f"blind-{key}-freeze.json"
    if receipt.is_file():
        if read_json(receipt) != current:
            raise ValueError("This blind split already has a different frozen run")
    elif create:
        write_json(receipt, current)
    else:
        raise ValueError("Freeze the qualified configuration before obtaining or scoring formal labels")
    return current


def verify_render(output, key, config):
    frozen = read_json(DEFAULT_OUTPUT / f"blind-{key}-freeze.json")
    if (frozen["output"] != str(output) or frozen["selected"]["config"] != config
            or frozen["inference_sha256"] != inference_hashes() or frozen["video_sha256"] != sha_file(VIDEO)):
        raise ValueError("The rendering inputs differ from the frozen blind configuration")

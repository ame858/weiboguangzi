"""Fetch missing pinned LabelMe files from public mirrors."""
import argparse
import subprocess
from pathlib import Path

from snl_inputs import BLIND, COMMIT, DEFAULT_OUTPUT, PROFILES, missing_labels
from snl_protocol import freeze_selection


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("split", choices=("development", *BLIND))
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--profile", choices=tuple(PROFILES), default="original")
    parser.add_argument("--missing-only", action="store_true", help="Print missing IDs without accessing the network")
    args = parser.parse_args()
    out = args.output.resolve()
    if args.split == "development":
        ids = tuple(i for split in PROFILES[args.profile] for i in split)
    else:
        freeze_selection(out, args.split)
        ids = tuple(i for block in BLIND[args.split].values() for i in block)
    missing = missing_labels(out, ids)
    if args.missing_only or not missing:
        print(f"{len(missing)} missing: {missing}")
        return
    labels = out / "labels"
    labels.mkdir(parents=True, exist_ok=True)
    mirrors = (
        f"https://raw.githubusercontent.com/Jiangxb8/Video-SAR/{COMMIT}/data/json",
        f"https://cdn.jsdelivr.net/gh/Jiangxb8/Video-SAR@{COMMIT}/data/json",
    )
    for swap in (0, 1):
        options = ["parallel", 'parallel-max = "8"', "fail", "location", "silent", "show-error",
                   'connect-timeout = "12"', 'max-time = "45"']
        for index in missing:
            name = f"{index:06}.json"
            mirror = mirrors[(index + swap) % 2]
            options += [f'url = "{mirror}/{name}"', f'output = "{(labels / name).as_posix()}"']
        config = out / f"{args.split}-downloads.txt"
        config.write_text("\n".join(options), encoding="utf-8")
        subprocess.run(["curl.exe", "--config", str(config)], check=False)
        missing = missing_labels(out, ids)
        if not missing:
            break
    print(f"{len(ids) - len(missing)}/{len(ids)} annotation files verified", flush=True)
    if missing:
        raise FileNotFoundError(f"Missing or corrupt Git blobs: {missing}")


if __name__ == "__main__":
    main()

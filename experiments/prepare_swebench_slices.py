from __future__ import annotations

import json
from pathlib import Path

from carve.datasets.io import iter_jsonl
from carve.datasets.swebench import validate_prepared_swebench_row


SLICE_NAMES = (
    "swebench_lite_slice00_000_099.jsonl",
    "swebench_lite_slice01_100_199.jsonl",
    "swebench_lite_slice02_200_299.jsonl",
)


def prepare_swebench_slices(input_path: str | Path, output_dir: str | Path, slice_size: int = 100) -> list[Path]:
    if slice_size != 100:
        raise ValueError("SWE-bench slices must contain exactly 100 cases")
    rows = list(iter_jsonl(input_path))
    for row_number, row in enumerate(rows):
        validate_prepared_swebench_row(row, row_number=row_number)
    if len(rows) != 300:
        raise ValueError(f"expected exactly 300 SWE-bench rows, found {len(rows)}")
    output_root = Path(output_dir)
    output_root.mkdir(parents=True, exist_ok=True)
    outputs: list[Path] = []
    for index, name in enumerate(SLICE_NAMES):
        output = output_root / name
        with output.open("w", encoding="utf-8") as handle:
            for row in rows[index * slice_size : (index + 1) * slice_size]:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        outputs.append(output)
    return outputs


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="data/raw/swebench_lite_prepared.jsonl")
    parser.add_argument("--output-dir", default="data/raw")
    args = parser.parse_args()
    outputs = prepare_swebench_slices(args.input, args.output_dir)
    print(json.dumps({"outputs": [str(path) for path in outputs], "rows_per_slice": 100}, indent=2))


if __name__ == "__main__":
    main()

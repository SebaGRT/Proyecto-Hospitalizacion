#!/usr/bin/env python3
from __future__ import annotations

import codecs
from pathlib import Path
from typing import Iterable, Tuple

import pandas as pd


DATASET_DIR = Path("DATASET-PROBLEMA8")
UTF8_DIR = DATASET_DIR / "utf8"
SAMPLE_SIZE = 5000
CHUNK_SIZE = 1024 * 1024  # 1 MB

FILES = [
    "GRD_PUBLICO_2019.csv",
    "GRD_PUBLICO_2020.csv",
    "GRD_PUBLICO_2021.csv",
    "GRD_PUBLICO_EXTERNO_2022.csv",
    "GRD_PUBLICO_2023.csv",
    "GRD_PUBLICO_2024.csv",
]

# Ordered by most likely; explicit UTF-16 variants avoid BOM-related failures.
CANDIDATE_ENCODINGS = [
    "utf-8-sig",
    "utf-8",
    "utf-16",
    "utf-16-le",
    "utf-16-be",
    "cp1252",
    "latin-1",
]


def convert_csv_to_utf8_stream(
    src_path: Path,
    dst_path: Path,
    encodings: Iterable[str] = CANDIDATE_ENCODINGS,
) -> str:
    """Convert a CSV file to UTF-8 using streaming decode/encode.

    This avoids loading the entire file into memory.
    Returns the encoding that worked.
    """
    dst_path.parent.mkdir(parents=True, exist_ok=True)

    for enc in encodings:
        tmp_path = dst_path.with_suffix(dst_path.suffix + ".tmp")
        try:
            decoder = codecs.getincrementaldecoder(enc)(errors="strict")
            with src_path.open("rb") as src, tmp_path.open("w", encoding="utf-8", newline="") as dst:
                while True:
                    raw = src.read(CHUNK_SIZE)
                    if not raw:
                        break
                    dst.write(decoder.decode(raw))
                dst.write(decoder.decode(b"", final=True))

            tmp_path.replace(dst_path)
            return enc
        except UnicodeDecodeError:
            if tmp_path.exists():
                tmp_path.unlink()
            continue

    # Last-resort: keep flow alive by replacing invalid bytes.
    fallback_tmp = dst_path.with_suffix(dst_path.suffix + ".tmp")
    decoder = codecs.getincrementaldecoder("utf-8")(errors="replace")
    with src_path.open("rb") as src, fallback_tmp.open("w", encoding="utf-8", newline="") as dst:
        while True:
            raw = src.read(CHUNK_SIZE)
            if not raw:
                break
            dst.write(decoder.decode(raw))
        dst.write(decoder.decode(b"", final=True))

    fallback_tmp.replace(dst_path)
    return "utf-8 (errors=replace)"


def sample_load_csv(path_utf8: Path, nrows: int = SAMPLE_SIZE) -> pd.DataFrame:
    """Load only a small sample from a UTF-8 CSV to reduce memory use."""
    return pd.read_csv(
        path_utf8,
        sep="|",
        nrows=nrows,
        engine="python",
        on_bad_lines="skip",
        encoding="utf-8",
        low_memory=True,
    )


def run_smoke_test() -> Tuple[int, int]:
    ok = 0
    fail = 0

    print(f"Running sample load test with nrows={SAMPLE_SIZE} per file")
    print(f"Dataset dir: {DATASET_DIR.resolve()}")

    for name in FILES:
        src = DATASET_DIR / name
        dst = UTF8_DIR / name

        if not src.exists():
            fail += 1
            print(f"[MISSING] {src}")
            continue

        try:
            used_enc = convert_csv_to_utf8_stream(src, dst)
            sample = sample_load_csv(dst, nrows=SAMPLE_SIZE)
            ok += 1
            print(
                f"[OK] {name} | detected={used_enc} | rows={len(sample)} | cols={sample.shape[1]}"
            )
        except Exception as exc:
            fail += 1
            print(f"[FAIL] {name} | {type(exc).__name__}: {exc}")

    return ok, fail


if __name__ == "__main__":
    ok_count, fail_count = run_smoke_test()
    print("-" * 80)
    print(f"Summary: OK={ok_count} FAIL={fail_count}")

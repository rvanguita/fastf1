"""Compara leituras Delta completas e filtradas por temporada."""

from __future__ import annotations

import argparse
import os
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from time import perf_counter

from deltalake import DeltaTable
from deltalake.exceptions import DeltaError

REPO_ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class ReadProfile:
    rows: int
    bytes: int
    seconds: float


def profile_read(path: str, filters: list[tuple] | None = None) -> ReadProfile:
    started = perf_counter()
    table = DeltaTable(path).to_pyarrow_table(filters=filters)
    return ReadProfile(
        rows=table.num_rows,
        bytes=table.nbytes,
        seconds=perf_counter() - started,
    )


def latest_season(bronze_path: str) -> int:
    years = DeltaTable(bronze_path).to_pyarrow_table(columns=["Year"])["Year"]
    return int(max(years.to_pylist()))


def print_comparison(
    name: str, full: ReadProfile, filtered: ReadProfile, max_ratio: float
) -> bool:
    ratio = filtered.bytes / full.bytes if full.bytes else 0.0
    print(
        f"{name:<8} "
        f"full={full.rows:>6} rows/{full.bytes / 1_000_000:>6.2f} MB/"
        f"{full.seconds:>6.3f}s  "
        f"season={filtered.rows:>5} rows/{filtered.bytes / 1_000_000:>6.2f} MB/"
        f"{filtered.seconds:>6.3f}s  ratio={ratio:>6.1%}"
    )
    return ratio <= max_ratio


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--bronze",
        default=os.getenv("TABLE_PATH_BRONZE", REPO_ROOT / "data/bronze/results"),
    )
    parser.add_argument(
        "--abt",
        default=os.getenv("TABLE_PATH_SILVER", REPO_ROOT / "data/silver/tb_abt"),
    )
    parser.add_argument("--year", type=int)
    parser.add_argument("--max-byte-ratio", type=float, default=0.10)
    args = parser.parse_args()

    year = args.year or latest_season(str(args.bronze))
    print(f"Temporada: {year} · limite de bytes: {args.max_byte_ratio:.0%}")

    checks = []
    bronze_full = profile_read(str(args.bronze))
    bronze_year = profile_read(str(args.bronze), [("Year", "=", year)])
    checks.append(
        print_comparison("Bronze", bronze_full, bronze_year, args.max_byte_ratio)
    )

    try:
        abt_full = profile_read(str(args.abt))
        abt_year = profile_read(
            str(args.abt),
            [
                ("dt_ref", ">=", date(year, 1, 1)),
                ("dt_ref", "<", date(year + 1, 1, 1)),
            ],
        )
        checks.append(print_comparison("ABT", abt_full, abt_year, args.max_byte_ratio))
    except (DeltaError, OSError):
        print("ABT      indisponível; benchmark preditivo ignorado")

    if not all(checks):
        raise SystemExit("Uma leitura sazonal excedeu o limite de bytes")


if __name__ == "__main__":
    main()

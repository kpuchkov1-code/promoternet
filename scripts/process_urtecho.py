"""Read raw Urtecho 2019 files, build the tidy dataset, and cache to parquet."""

from __future__ import annotations

import sys

from promoternet.data import cache_processed, load_urtecho_2019


def main() -> int:
    print("Loading raw Urtecho 2019 files ...")
    df = load_urtecho_2019()
    print(f"  loaded {len(df):,} rows")
    print("\n  schema:")
    print(df.dtypes.to_string())
    print("\n  head:")
    print(df.head().to_string())
    print("\n  expression summary:")
    print(df[["expression", "log_expression"]].describe().to_string())
    print("\n  element class counts:")
    for col in ("up_class", "m35_pattern", "spacer", "m10_pattern", "background"):
        nunique = df[col].nunique()
        top = df[col].value_counts().head(3).to_dict()
        print(f"    {col}: {nunique} unique. top: {top}")

    out = cache_processed(df, "urtecho2019")
    print(f"\n  cached to: {out} ({out.stat().st_size:,} bytes)")
    return 0


if __name__ == "__main__":
    sys.exit(main())

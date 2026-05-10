"""Download bacterial promoter datasets used to train and evaluate PromoterNet.

Primary dataset: Urtecho et al 2019 (Biochemistry) — 10,898 σ70 promoter variants
from a combinatorial library (8 × -35 × 8 × -10 × 3 UP × 8 spacers × 8 backgrounds).
Hosted publicly on the Kosuri Lab GitHub at
https://github.com/KosuriLab/ecoli_minimal_promoter — no auth required.

Stretch dataset: Kosuri et al 2013 PNAS — 12,563 promoter+RBS combinations.
Supplementary tables hosted on the PNAS website. May require manual download
because PNAS has anti-bot protection on supplementary file downloads.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

import requests
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = PROJECT_ROOT / "data" / "raw"
RAW_DIR.mkdir(parents=True, exist_ok=True)


@dataclass(frozen=True)
class DatasetFile:
    name: str
    url: str
    target: Path
    sha256: str | None = None
    description: str = ""


URTECHO_2019_FILES: list[DatasetFile] = [
    DatasetFile(
        name="urtecho_variant_statistics",
        url="https://raw.githubusercontent.com/KosuriLab/ecoli_minimal_promoter/master/processed_data/variant_statistics.txt",
        target=RAW_DIR / "urtecho2019" / "variant_statistics.txt",
        description="Per-variant RNA/DNA ratios + element identifiers for 10,898 σ70 variants.",
    ),
    DatasetFile(
        name="urtecho_expression",
        url="https://raw.githubusercontent.com/KosuriLab/ecoli_minimal_promoter/master/processed_data/revLP5_Min_MOPS_glu_expression.txt",
        target=RAW_DIR / "urtecho2019" / "revLP5_Min_MOPS_glu_expression.txt",
        description="Expression measurements (MOPS+glucose, replicate-aggregated).",
    ),
    DatasetFile(
        name="urtecho_library_ref",
        url="https://raw.githubusercontent.com/KosuriLab/ecoli_minimal_promoter/master/ref/min_promoter_lib_up_20161118_controls_clean.txt",
        target=RAW_DIR / "urtecho2019" / "library_reference.txt",
        description="Library design reference: per-variant element identifiers + full promoter sequence.",
    ),
]


def stream_download(url: str, target: Path, chunk_size: int = 1 << 14) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists() and target.stat().st_size > 0:
        print(f"  skip (exists): {target.name} ({target.stat().st_size:,} bytes)")
        return

    with requests.get(url, stream=True, timeout=60) as response:
        response.raise_for_status()
        total = int(response.headers.get("Content-Length", 0))
        with (
            open(target, "wb") as fh,
            tqdm(
                total=total or None,
                unit="B",
                unit_scale=True,
                unit_divisor=1024,
                desc=target.name,
                leave=False,
            ) as bar,
        ):
            for chunk in response.iter_content(chunk_size=chunk_size):
                if not chunk:
                    continue
                fh.write(chunk)
                bar.update(len(chunk))


def download_all(files: list[DatasetFile]) -> list[Path]:
    written: list[Path] = []
    for f in files:
        print(f"\n--- {f.name} ---")
        print(f"  url: {f.url}")
        try:
            stream_download(f.url, f.target)
            written.append(f.target)
            print(f"  ok: {f.target} ({f.target.stat().st_size:,} bytes)")
        except requests.HTTPError as e:
            print(f"  FAILED HTTP {e.response.status_code}: {e}", file=sys.stderr)
        except Exception as e:
            print(f"  FAILED: {e}", file=sys.stderr)
    return written


def head_preview(path: Path, n: int = 5) -> None:
    print(f"\n  head ({n} lines) of {path.name}:")
    with open(path, encoding="utf-8", errors="replace") as fh:
        for i, line in enumerate(fh):
            if i >= n:
                break
            print(f"    {line.rstrip()[:200]}")


def main() -> int:
    print("PromoterNet data download")
    print("=" * 60)
    written = download_all(URTECHO_2019_FILES)

    if not written:
        print("\nNo files downloaded.", file=sys.stderr)
        return 1

    print("\nFile previews:")
    for path in written:
        head_preview(path, n=5)

    print(
        f"\nDone. {len(written)}/{len(URTECHO_2019_FILES)} files in {RAW_DIR / 'urtecho2019'}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())

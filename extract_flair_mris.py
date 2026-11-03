from __future__ import annotations

import argparse
import gzip
import shutil
import tarfile
import zipfile
from pathlib import Path


def is_flair_file(name: str) -> bool:
    lower_name = name.lower()
    return lower_name.endswith("_flair.nii") or lower_name.endswith("_flair.nii.gz")


def output_name_for(source_name: str) -> str:
    lower_name = source_name.lower()
    if lower_name.endswith(".nii.gz"):
        return source_name[:-3]
    return source_name


def write_nifti_bytes(source_stream, destination: Path, compressed: bool) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if compressed:
        with gzip.GzipFile(fileobj=source_stream, mode="rb") as compressed_stream, destination.open("wb") as target_stream:
            shutil.copyfileobj(compressed_stream, target_stream)
    else:
        with destination.open("wb") as target_stream:
            shutil.copyfileobj(source_stream, target_stream)


def copy_file(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)


def extract_from_zip(archive_path: Path, output_dir: Path, seen: set[str]) -> int:
    extracted = 0
    with zipfile.ZipFile(archive_path) as archive:
        for member in archive.infolist():
            if member.is_dir():
                continue
            member_name = Path(member.filename).name
            if not is_flair_file(member_name):
                continue

            destination_name = output_name_for(member_name)
            if destination_name in seen or (output_dir / destination_name).exists():
                continue

            destination = output_dir / destination_name
            with archive.open(member) as source_stream:
                write_nifti_bytes(source_stream, destination, member_name.lower().endswith(".nii.gz"))
            seen.add(destination_name)
            extracted += 1
    return extracted


def extract_from_tar(archive_path: Path, output_dir: Path, seen: set[str]) -> int:
    extracted = 0
    with tarfile.open(archive_path) as archive:
        for member in archive.getmembers():
            if not member.isfile():
                continue
            member_name = Path(member.name).name
            if not is_flair_file(member_name):
                continue

            destination_name = output_name_for(member_name)
            if destination_name in seen or (output_dir / destination_name).exists():
                continue

            source_stream = archive.extractfile(member)
            if source_stream is None:
                continue

            destination = output_dir / destination_name
            with source_stream:
                write_nifti_bytes(source_stream, destination, member_name.lower().endswith(".nii.gz"))
            seen.add(destination_name)
            extracted += 1
    return extracted


def extract_flair_mris(source_root: Path, output_dir: Path) -> dict[str, int]:
    """Extract all unique FLAIR MRI scans into a flat output directory.

    The extractor walks patient folders recursively, handles compressed archives
    when present, decompresses .nii.gz files into .nii, and skips any output
    filename that already exists.
    """

    source_root = source_root.resolve()
    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    seen = {path.name for path in output_dir.iterdir() if path.is_file() and is_flair_file(path.name)}
    counts = {"copied": 0, "decompressed": 0, "archives": 0, "skipped": 0}

    for path in source_root.rglob("*"):
        if not path.is_file():
            continue

        lower_name = path.name.lower()

        if lower_name.endswith(".nii") and is_flair_file(path.name):
            destination_name = output_name_for(path.name)
            destination = output_dir / destination_name
            if destination_name in seen or destination.exists():
                counts["skipped"] += 1
                continue
            copy_file(path, destination)
            seen.add(destination_name)
            counts["copied"] += 1
            continue

        if lower_name.endswith(".nii.gz") and is_flair_file(path.name):
            destination_name = output_name_for(path.name)
            destination = output_dir / destination_name
            if destination_name in seen or destination.exists():
                counts["skipped"] += 1
                continue
            with gzip.open(path, "rb") as source_stream:
                write_nifti_bytes(source_stream, destination, False)
            seen.add(destination_name)
            counts["decompressed"] += 1
            continue

        if lower_name.endswith(".zip"):
            counts["archives"] += extract_from_zip(path, output_dir, seen)
            continue

        if lower_name.endswith((".tar", ".tar.gz", ".tgz", ".tbz2", ".txz")):
            counts["archives"] += extract_from_tar(path, output_dir, seen)

    counts["total_written"] = counts["copied"] + counts["decompressed"] + counts["archives"]
    return counts


def build_default_paths() -> tuple[Path, Path]:
    base_dir = Path(__file__).resolve().parent
    source_root = base_dir / "PKG - Yale-Brain-Mets-Longitudinal" / "Yale-Brain-Mets-Longitudinal"
    output_dir = base_dir / "dataset"
    return source_root, output_dir


def main() -> int:
    parser = argparse.ArgumentParser(description="Extract unique FLAIR MRI scans into a flat dataset folder.")
    parser.add_argument("--source-root", type=Path, default=None, help="Root folder containing patient records")
    parser.add_argument("--output-dir", type=Path, default=None, help="Folder where extracted FLAIR .nii files are saved")
    args = parser.parse_args()

    default_source_root, default_output_dir = build_default_paths()
    source_root = args.source_root or default_source_root
    output_dir = args.output_dir or default_output_dir

    if not source_root.exists():
        raise FileNotFoundError(f"Source root does not exist: {source_root}")

    counts = extract_flair_mris(source_root, output_dir)
    print(
        "Finished extracting FLAIR scans: "
        f"{counts['total_written']} new files written "
        f"({counts['copied']} copied, {counts['decompressed']} decompressed, {counts['archives']} from archives, {counts['skipped']} skipped as duplicates)."
    )
    print(f"Output folder: {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
#!/usr/bin/env python3
"""Download TTS datasets."""

import argparse
import os
import sys
import tarfile
import zipfile
import subprocess
import shutil
from pathlib import Path
from urllib.request import urlretrieve
from tqdm import tqdm


class DownloadProgress:
    """Progress bar for downloads."""

    def __init__(self, desc: str):
        self.pbar = None
        self.desc = desc

    def __call__(self, block_num, block_size, total_size):
        if self.pbar is None:
            self.pbar = tqdm(
                total=total_size,
                unit="B",
                unit_scale=True,
                desc=self.desc,
            )
        downloaded = block_num * block_size
        self.pbar.update(downloaded - self.pbar.n)

    def close(self):
        if self.pbar:
            self.pbar.close()


def _is_within_directory(base: Path, target: Path) -> bool:
    base = base.resolve()
    target = target.resolve()
    return str(target).startswith(str(base) + os.sep)


def safe_extract_tar(tar: tarfile.TarFile, path: Path) -> None:
    """Safely extract tar files to avoid path traversal."""
    for member in tar.getmembers():
        member_path = path / member.name
        if not _is_within_directory(path, member_path):
            raise RuntimeError(f"Blocked path traversal attempt: {member.name}")
    tar.extractall(path)


def safe_extract_zip(zip_ref: zipfile.ZipFile, path: Path) -> None:
    """Safely extract zip files to avoid path traversal."""
    for member in zip_ref.namelist():
        member_path = path / member
        if not _is_within_directory(path, member_path):
            raise RuntimeError(f"Blocked path traversal attempt: {member}")
    zip_ref.extractall(path)


def download_ljspeech(output_dir: Path) -> None:
    """Download LJSpeech dataset.

    Args:
        output_dir: Output directory
    """
    url = "https://data.keithito.com/data/speech/LJSpeech-1.1.tar.bz2"
    tar_path = output_dir / "LJSpeech-1.1.tar.bz2"

    output_dir.mkdir(parents=True, exist_ok=True)

    print("Downloading LJSpeech...")
    progress = DownloadProgress("LJSpeech")
    urlretrieve(url, tar_path, progress)
    progress.close()

    print("Extracting...")
    with tarfile.open(tar_path, "r:bz2") as tar:
        safe_extract_tar(tar, output_dir)

    tar_path.unlink()
    print(f"LJSpeech downloaded to {output_dir / 'LJSpeech-1.1'}")


def download_vctk(output_dir: Path) -> None:
    """Download VCTK dataset.

    Args:
        output_dir: Output directory
    """
    url = "https://datashare.ed.ac.uk/bitstream/handle/10283/3443/VCTK-Corpus-0.92.zip"
    zip_path = output_dir / "VCTK-Corpus-0.92.zip"

    output_dir.mkdir(parents=True, exist_ok=True)

    print("Downloading VCTK (this is a large file, ~11GB)...")
    print("Note: You may need to download manually from:")
    print("  https://datashare.ed.ac.uk/handle/10283/3443")

    try:
        if shutil.which("aria2c"):
            print("Using aria2c for faster download...")
            cmd = [
                "aria2c",
                "-x", "16",
                "-s", "16",
                "-c",
                "-o", zip_path.name,
                url,
            ]
            try:
                subprocess.run(cmd, check=True, cwd=str(output_dir))
            except subprocess.CalledProcessError:
                print("aria2c multi-connection failed; retrying single connection...")
                cmd = [
                    "aria2c",
                    "-x", "1",
                    "-s", "1",
                    "-c",
                    "--file-allocation=none",
                    "-o", zip_path.name,
                    url,
                ]
                subprocess.run(cmd, check=True, cwd=str(output_dir))
        else:
            progress = DownloadProgress("VCTK")
            urlretrieve(url, zip_path, progress)
            progress.close()

        print("Extracting...")
        with zipfile.ZipFile(zip_path, "r") as zip_ref:
            safe_extract_zip(zip_ref, output_dir)

        zip_path.unlink()
        print(f"VCTK downloaded to {output_dir}")
    except Exception as e:
        print(f"Download failed: {e}")
        print("Please download manually from the URL above.")


def download_libritts(output_dir: Path, subsets: list) -> None:
    """Download LibriTTS dataset.

    Args:
        output_dir: Output directory
        subsets: List of subsets to download
    """
    base_url = "https://www.openslr.org/resources/60"

    output_dir.mkdir(parents=True, exist_ok=True)

    for subset in subsets:
        tar_name = f"{subset}.tar.gz"
        url = f"{base_url}/{tar_name}"
        tar_path = output_dir / tar_name

        print(f"Downloading LibriTTS {subset}...")
        progress = DownloadProgress(subset)
        try:
            urlretrieve(url, tar_path, progress)
            progress.close()

            print(f"Extracting {subset}...")
            with tarfile.open(tar_path, "r:gz") as tar:
                safe_extract_tar(tar, output_dir)

            tar_path.unlink()
            print(f"  {subset} extracted")
        except Exception as e:
            print(f"  Failed to download {subset}: {e}")
            progress.close()

    print(f"LibriTTS downloaded to {output_dir}")


def main():
    parser = argparse.ArgumentParser(description="Download TTS datasets")
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        choices=["ljspeech", "vctk", "libritts", "all"],
        help="Dataset to download",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./data",
        help="Output directory",
    )
    parser.add_argument(
        "--libritts-subsets",
        type=str,
        nargs="+",
        default=["train-clean-100"],
        help="LibriTTS subsets to download",
    )

    args = parser.parse_args()
    output_dir = Path(args.output_dir)

    if args.dataset == "ljspeech" or args.dataset == "all":
        download_ljspeech(output_dir)

    if args.dataset == "vctk" or args.dataset == "all":
        download_vctk(output_dir)

    if args.dataset == "libritts" or args.dataset == "all":
        download_libritts(output_dir / "LibriTTS", args.libritts_subsets)

    print("Done!")


if __name__ == "__main__":
    main()

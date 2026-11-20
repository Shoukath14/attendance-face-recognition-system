"""One-time model download script and offline verification."""

import argparse
import sys
import zipfile
from pathlib import Path

import requests
import urllib3

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

BUFFALO_S_URL = "https://github.com/deepinsight/insightface/releases/download/model-zoo/buffalo_s.zip"
BUFFALO_L_URL = "https://github.com/deepinsight/insightface/releases/download/model-zoo/buffalo_l.zip"


def download_zip(url: str, dest_path: Path):
    """Download file handling GitHub redirect cleanly without proxy issues."""
    print(f"Resolving download URL: {url}")
    dest_path.parent.mkdir(parents=True, exist_ok=True)

    # Step 1: Resolve redirect without following immediately
    try:
        r1 = requests.get(url, allow_redirects=False, verify=False, timeout=30)
    except Exception as e:
        print(f"Error connecting to release URL: {e}")
        raise

    target_url = r1.headers.get("Location", url)
    print("Direct asset URL resolved. Downloading payload...")

    # Step 2: Download the binary directly
    r2 = requests.get(target_url, stream=True, verify=False, timeout=120)
    r2.raise_for_status()

    total_size = int(r2.headers.get("content-length", 0))
    chunk_size = 2 * 1024 * 1024
    downloaded = 0

    with open(dest_path, "wb") as f:
        for chunk in r2.iter_content(chunk_size=chunk_size):
            if chunk:
                f.write(chunk)
                downloaded += len(chunk)
                if total_size > 0:
                    percent = (downloaded / total_size) * 100
                    print(
                        f"\r  Progress: {downloaded / (1024*1024):.1f} MB / {total_size / (1024*1024):.1f} MB ({percent:.1f}%)",
                        end="",
                        flush=True,
                    )
    print("\nDownload complete!")


def setup_insightface_pack(model_pack: str = "buffalo_s", target_dirs: list[Path] = None) -> bool:
    """Download and extract insightface pack into target directories."""
    url = BUFFALO_S_URL if model_pack == "buffalo_s" else BUFFALO_L_URL
    temp_zip = Path("models") / f"{model_pack}.zip"

    default_insightface_home = Path.home() / ".insightface" / "models" / model_pack
    dirs_to_populate = [default_insightface_home]
    if target_dirs:
        dirs_to_populate.extend(target_dirs)

    # Check if already present
    all_exist = all(d.exists() and any(d.glob("*.onnx")) for d in dirs_to_populate)
    if not all_exist:
        if not temp_zip.exists():
            download_zip(url, temp_zip)

        for dest_dir in dirs_to_populate:
            print(f"Extracting to {dest_dir}...")
            dest_dir.mkdir(parents=True, exist_ok=True)
            with zipfile.ZipFile(temp_zip, "r") as zf:
                zf.extractall(dest_dir)
            onnx_files = list(dest_dir.glob("*.onnx"))
            print(f"Extracted {len(onnx_files)} ONNX models to {dest_dir}")
            for m in onnx_files:
                print(f"  - {m.name} ({m.stat().st_size / (1024*1024):.2f} MB)")

        if temp_zip.exists():
            temp_zip.unlink()

    # Now verify offline loading
    print("\nVerifying offline load with InsightFace FaceAnalysis...")
    try:
        from insightface.app import FaceAnalysis

        app = FaceAnalysis(name=model_pack, providers=["CPUExecutionProvider"])
        app.prepare(ctx_id=0, det_size=(640, 640))
        print("Successfully loaded components:", list(app.models.keys()))
        print("Offline model verification PASSED!")
        return True
    except Exception as e:
        print(f"Failed to load models in FaceAnalysis: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(description="Download and verify models")
    parser.add_argument("--pack", type=str, default="buffalo_s", help="Model pack name")
    args = parser.parse_args()

    local_models_dir = Path("models") / "models" / args.pack
    success = setup_insightface_pack(args.pack, [local_models_dir])
    if not success:
        sys.exit(1)


if __name__ == "__main__":
    main()

"""Probe camera indices 0-5 and display available video sources and resolutions."""

import sys

import cv2


def probe_cameras(max_index: int = 5) -> list[dict]:
    """Probe camera devices from index 0 up to max_index."""
    found = []
    print(f"Probing video capture devices (indices 0 to {max_index})...\n")

    for idx in range(max_index + 1):
        cap = cv2.VideoCapture(idx, cv2.CAP_DSHOW if sys.platform.startswith("win") else cv2.CAP_ANY)
        if cap.isOpened():
            ret, frame = cap.read()
            if ret and frame is not None:
                width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                fps = cap.get(cv2.CAP_PROP_FPS)
                found.append({"index": idx, "width": width, "height": height, "fps": fps})
                print(f"  [FOUND] Camera Index {idx}: Resolution {width}x{height}, FPS {fps:.1f}")
            else:
                print(f"  [OPENED BUT NO FRAME] Camera Index {idx}")
            cap.release()
        else:
            print(f"  [NOT FOUND] Camera Index {idx}")

    return found


def main():
    cameras = probe_cameras(5)
    print("\nSummary:")
    if not cameras:
        print("  No working cameras found. Please check your webcam or phone connection (DroidCam/Iriun).")
    else:
        print(f"  Found {len(cameras)} working camera(s):")
        for cam in cameras:
            print(f"    - Index {cam['index']}: {cam['width']}x{cam['height']} @ {cam['fps']:.1f} FPS")
        print("\nTo use one in config.yaml, set:")
        print(f"  camera:\n    source: {cameras[0]['index']}")


if __name__ == "__main__":
    main()


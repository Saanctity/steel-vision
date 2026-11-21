"""Publish the simulated camera videos as RTSP streams (needs MediaMTX running).

Each video loops forever and is re-encoded to H.264, like a real IP camera.
Press Ctrl+C to stop all streams.

Usage (from repo root):
    python -m ingest.stream_videos --cameras 4
"""
from __future__ import annotations

import argparse
import subprocess
import time
from pathlib import Path


def ffmpeg_cmd(ffmpeg: str, video: Path, url: str, fps: int = 25) -> list[str]:
    return [ffmpeg, "-hide_banner", "-loglevel", "error",
            "-re", "-stream_loop", "-1", "-i", str(video),   # -re: play at real time
            "-an", "-c:v", "libx264", "-preset", "ultrafast", "-tune", "zerolatency",
            "-pix_fmt", "yuv420p", "-g", str(fps),
            "-f", "rtsp", "-rtsp_transport", "tcp", url]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cameras", type=int, default=4)
    ap.add_argument("--videos", type=Path, default=Path("videos"))
    ap.add_argument("--ffmpeg", default="tools/ffmpeg.exe" if Path("tools/ffmpeg.exe").exists() else "ffmpeg")
    ap.add_argument("--host", default="localhost")
    ap.add_argument("--port", type=int, default=8554)
    ap.add_argument("--fps", type=int, default=25)
    args = ap.parse_args()

    procs = []
    for cam in range(1, args.cameras + 1):
        video = args.videos / f"cam{cam}.mp4"
        if not video.exists():
            raise SystemExit(f"Missing {video}. Run: python -m ingest.make_strip_video")
        url = f"rtsp://{args.host}:{args.port}/cam{cam}"
        procs.append(subprocess.Popen(ffmpeg_cmd(args.ffmpeg, video, url, args.fps)))
        print(f"streaming {video} -> {url}")
    print("Streams running. Ctrl+C to stop.")
    try:
        while True:
            time.sleep(1)
            dead = [i + 1 for i, p in enumerate(procs) if p.poll() is not None]
            if dead:
                print(f"ffmpeg for camera(s) {dead} exited - is MediaMTX running?")
                break
    except KeyboardInterrupt:
        pass
    finally:
        for p in procs:
            p.terminate()


if __name__ == "__main__":
    main()
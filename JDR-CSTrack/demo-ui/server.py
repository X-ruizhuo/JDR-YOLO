"""Local video analysis server for the JDR-CSTrack demonstration."""

import argparse
import cgi
import json
import os
import shutil
import subprocess
import threading
import time
import uuid
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse

ROOT = Path(__file__).resolve().parent
PROJECT = ROOT.parent
TRACKER = PROJECT / "ultralytics/cfg/trackers/cstrack.yaml"
RUNS = ROOT / "runs"
MAX_UPLOAD = 1024 * 1024 * 1024
jobs = {}
lock = threading.Lock()


def analyze(job_id, source, weights, conf):
    job = jobs[job_id]
    try:
        import cv2
        from ultralytics import YOLO

        capture = cv2.VideoCapture(str(source))
        if not capture.isOpened():
            raise ValueError("无法打开视频文件")
        fps = capture.get(cv2.CAP_PROP_FPS) or 25
        width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        if width < 1 or height < 1:
            raise ValueError("视频分辨率无效")
        job["total"] = total
        job["fps"] = round(fps, 2)
        job["resolution"] = f"{width} x {height}"
        raw = source.parent / "raw.mp4"
        writer = cv2.VideoWriter(str(raw), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
        if not writer.isOpened():
            raise RuntimeError("无法创建输出视频")
        model = YOLO(str(weights))
        frames = []
        unique = set()
        classes = set()
        started = time.monotonic()
        try:
            index = 0
            while True:
                ok, frame = capture.read()
                if not ok:
                    break
                result = model.track(frame, persist=True, tracker=str(TRACKER), conf=conf,
                                     imgsz=640, device=0, verbose=False)[0]
                objects = []
                for box in result.boxes:
                    track_id = int(box.id.item()) if box.id is not None else None
                    cls = int(box.cls.item())
                    if track_id is not None:
                        unique.add(track_id)
                    classes.add(cls)
                    objects.append({"id": track_id, "class": result.names[cls],
                                    "confidence": round(float(box.conf.item()), 3),
                                    "box": [round(float(v), 1) for v in box.xyxy[0].tolist()]})
                writer.write(result.plot())
                frames.append({"time": round(index / fps, 3), "objects": objects})
                index += 1
                job["processed"] = index
            if not frames:
                raise ValueError("视频没有可读取的帧")
        finally:
            writer.release()
            capture.release()

        output = source.parent / "result.mp4"
        if shutil.which("ffmpeg"):
            subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", str(raw),
                            "-c:v", "libx264", "-pix_fmt", "yuv420p", "-movflags", "+faststart",
                            str(output)], check=True)
            raw.unlink()
        else:
            raw.rename(output)
        payload = {"frames": frames, "summary": {"frames": len(frames),
                   "unique_tracks": len(unique), "classes": len(classes),
                   "elapsed_seconds": round(time.monotonic() - started, 1),
                   "processing_fps": round(len(frames) / max(time.monotonic() - started, .001), 1)}}
        (source.parent / "results.json").write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
        job["status"] = "done"
    except Exception as exc:
        job["status"] = "error"
        job["error"] = str(exc)


class Handler(SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=str(ROOT), **kwargs)

    def do_GET(self):
        path = urlparse(self.path).path
        if path.startswith("/api/jobs/"):
            job_id = path.removeprefix("/api/jobs/")
            job = jobs.get(job_id)
            if not job:
                return self.send_json({"error": "任务不存在"}, 404)
            data = dict(job)
            if job["status"] == "done":
                data["result"] = f"/runs/{job_id}/results.json"
                data["video"] = f"/runs/{job_id}/result.mp4"
            return self.send_json(data)
        return super().do_GET()

    def do_POST(self):
        if urlparse(self.path).path != "/api/jobs":
            return self.send_json({"error": "接口不存在"}, 404)
        length = int(self.headers.get("Content-Length", "0"))
        if length < 1 or length > MAX_UPLOAD:
            return self.send_json({"error": "视频文件不能为空或超过 1 GB"}, 413)
        if "multipart/form-data" not in self.headers.get("Content-Type", ""):
            return self.send_json({"error": "请上传视频文件"}, 400)
        field = cgi.FieldStorage(fp=self.rfile, headers=self.headers,
                                 environ={"REQUEST_METHOD": "POST", "CONTENT_TYPE": self.headers["Content-Type"]})
        if "video" not in field:
            return self.send_json({"error": "缺少视频文件"}, 400)
        video = field["video"]
        if isinstance(video, list) or not video.filename:
            return self.send_json({"error": "请选择视频文件"}, 400)
        if Path(video.filename).suffix.lower() not in {".mp4", ".mov", ".avi", ".mkv"}:
            return self.send_json({"error": "仅支持 MP4、MOV、AVI、MKV"}, 400)
        if not self.server.weights.is_file():
            return self.send_json({"error": f"模型尚未就绪: {self.server.weights}"}, 503)
        if any(job["status"] == "running" for job in jobs.values()):
            return self.send_json({"error": "已有任务正在处理，请等待完成"}, 409)
        job_id = uuid.uuid4().hex[:12]
        folder = RUNS / job_id
        folder.mkdir(parents=True)
        source = folder / ("source" + Path(video.filename).suffix.lower())
        with source.open("wb") as target:
            shutil.copyfileobj(video.file, target)
        jobs[job_id] = {"id": job_id, "status": "running", "processed": 0, "total": 0}
        threading.Thread(target=analyze, args=(job_id, source, self.server.weights, self.server.conf), daemon=True).start()
        self.send_json(jobs[job_id], 202)

    def send_json(self, data, status=200):
        body = json.dumps(data, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


def main():
    parser = argparse.ArgumentParser(description="JDR-CSTrack video demonstration")
    parser.add_argument("--weights", type=Path,
                        default=PROJECT / "runs/detect/shipsmot_half_v8l_b2/weights/best.pt",
                        help="Trained best.pt")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--conf", type=float, default=0.25)
    args = parser.parse_args()
    weights = args.weights.expanduser().resolve()
    if not 0 < args.conf < 1:
        parser.error("--conf 必须介于 0 和 1 之间")
    os.environ.setdefault("TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD", "1")
    RUNS.mkdir(exist_ok=True)
    server = ThreadingHTTPServer((args.host, args.port), Handler)
    server.weights, server.conf = weights, args.conf
    print(f"打开 http://{args.host}:{args.port}", flush=True)
    server.serve_forever()


if __name__ == "__main__":
    main()

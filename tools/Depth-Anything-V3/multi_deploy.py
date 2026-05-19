import io
import os
import queue
import tempfile
import time
from contextlib import asynccontextmanager
from typing import Optional

import matplotlib
import numpy as np
import torch
import torch.multiprocessing as mp
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse
from PIL import Image
from pydantic import BaseModel

from depth_anything_3.api import DepthAnything3

mp.set_start_method('spawn', force=True)

worker_pool = []
worker_index = 0

# Default to monocular relative depth (V2 drop-in). Set to depth-anything/DA3METRIC-LARGE
# via env var when metric depth is needed.
DEFAULT_MODEL_ID = os.environ.get("DA3_MODEL_ID", "depth-anything/DA3MONO-LARGE")


@asynccontextmanager
async def lifespan(app: FastAPI):
    global worker_pool, worker_index

    gpu_count = torch.cuda.device_count()
    if gpu_count == 0:
        raise RuntimeError("No GPU devices available")

    print(f"Detected {gpu_count} GPU devices")

    for gpu_id in range(gpu_count):
        worker = ModelWorker(gpu_id=gpu_id, model_id=DEFAULT_MODEL_ID)
        worker.start()
        worker_pool.append(worker)

    print(f"Initialized {len(worker_pool)} worker(s) for {DEFAULT_MODEL_ID}")

    yield

    print("Shutting down all model worker processes")
    for worker in worker_pool:
        worker.stop()
    worker_pool = []


app = FastAPI(
    title="Depth Anything 3 API",
    description="API for Depth Anything 3 depth prediction (monocular by default).",
    version="3.0.0",
    lifespan=lifespan,
)


class DepthResponse(BaseModel):
    depth_array: list
    width: int
    height: int
    min_depth: float
    max_depth: float


class ModelWorker:
    def __init__(self, gpu_id: int, model_id: str = DEFAULT_MODEL_ID):
        self.gpu_id = gpu_id
        self.model_id = model_id
        self.device = f'cuda:{gpu_id}'
        self.request_queue = mp.Queue()
        self.result_queue = mp.Queue()
        self.process: Optional[mp.Process] = None

    def start(self):
        self.process = mp.Process(
            target=self._worker_process,
            args=(self.gpu_id, self.model_id, self.request_queue, self.result_queue),
        )
        self.process.daemon = True
        self.process.start()

    def _worker_process(self, gpu_id, model_id, request_queue, result_queue):
        try:
            torch.cuda.set_device(gpu_id)
            device = torch.device(f'cuda:{gpu_id}')

            model = DepthAnything3.from_pretrained(model_id)
            model = model.to(device=device)

            cmap = matplotlib.colormaps.get_cmap('Spectral_r')
            print(f"DA3 model loaded on GPU:{gpu_id} ({model_id})")

            request_count = 0
            clean_interval = 100

            while True:
                request_id = None
                try:
                    request_id, image_bytes, request_type = request_queue.get()

                    # DA3 inference takes file paths; persist the upload to a temp file.
                    with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp:
                        tmp.write(image_bytes)
                        tmp_path = tmp.name

                    try:
                        with torch.no_grad():
                            prediction = model.inference([tmp_path])
                        depth = prediction.depth[0]
                        if isinstance(depth, torch.Tensor):
                            depth = depth.detach().cpu().numpy()
                    finally:
                        try:
                            os.unlink(tmp_path)
                        except OSError:
                            pass

                    if request_type == "raw":
                        result = {
                            "depth": depth,
                            "min_depth": float(depth.min()),
                            "max_depth": float(depth.max()),
                        }
                    elif request_type == "gray":
                        span = depth.max() - depth.min()
                        depth_norm = (depth - depth.min()) / (span + 1e-9) * 255.0
                        result = {"depth_norm": depth_norm.astype(np.uint8), "mode": "L"}
                    elif request_type == "color":
                        span = depth.max() - depth.min()
                        depth_norm = (depth - depth.min()) / (span + 1e-9) * 255.0
                        depth_norm = depth_norm.astype(np.uint8)
                        colored = (cmap(depth_norm)[..., :3] * 255).astype(np.uint8)
                        result = {"colored_depth": colored}
                    else:
                        result = {"error": f"unknown request_type {request_type!r}"}

                    result_queue.put((request_id, result))

                    request_count += 1
                    if request_count % clean_interval == 0:
                        torch.cuda.empty_cache()

                except Exception as e:
                    print(f"GPU {gpu_id} worker error: {e}")
                    result_queue.put((request_id, {"error": str(e)}))

        except Exception as e:
            print(f"GPU {gpu_id} initialization failed: {e}")

    def process_image(self, image_bytes: bytes, request_type: str) -> dict:
        request_id = f"{time.time()}_{id(image_bytes)}"
        self.request_queue.put((request_id, image_bytes, request_type))

        while True:
            try:
                result_id, result = self.result_queue.get(timeout=30)
                if result_id == request_id:
                    return result
            except queue.Empty:
                raise TimeoutError("Request processing timed out")

    def stop(self):
        if self.process and self.process.is_alive():
            self.process.terminate()
            self.process.join()


def get_next_worker():
    global worker_index, worker_pool
    if not worker_pool:
        raise RuntimeError("No available model worker processes")
    worker = worker_pool[worker_index]
    worker_index = (worker_index + 1) % len(worker_pool)
    return worker


async def _consume_upload(file: UploadFile) -> bytes:
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Uploaded file must be an image")
    return await file.read()


@app.post("/predict/color_depth")
async def predict_color_depth(file: UploadFile = File(...)):
    """Upload an image, return a colorized depth map PNG (Spectral_r)."""
    contents = await _consume_upload(file)
    try:
        result = get_next_worker().process_image(contents, request_type="color")
        if "error" in result:
            raise Exception(result["error"])

        colored_pil = Image.fromarray(result["colored_depth"])
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
            colored_pil.save(tmp.name)
            return FileResponse(tmp.name, media_type="image/png", filename="depth_map.png")

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing image: {e}")


@app.post("/predict/raw_depth_array")
async def predict_raw_array(file: UploadFile = File(...)):
    """Upload an image, return the raw depth array as JSON."""
    contents = await _consume_upload(file)
    try:
        result = get_next_worker().process_image(contents, request_type="raw")
        if "error" in result:
            raise Exception(result["error"])

        depth = result["depth"]
        return DepthResponse(
            depth_array=depth.flatten().tolist(),
            width=int(depth.shape[1]),
            height=int(depth.shape[0]),
            min_depth=result["min_depth"],
            max_depth=result["max_depth"],
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing image: {e}")


@app.post("/predict/depth")
async def predict_depth_map(file: UploadFile = File(...)):
    """Upload an image, return a grayscale depth map PNG."""
    contents = await _consume_upload(file)
    try:
        result = get_next_worker().process_image(contents, request_type="gray")
        if "error" in result:
            raise Exception(result["error"])

        depth_image = Image.fromarray(result["depth_norm"], mode=result["mode"])
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
            depth_image.save(tmp.name)
            return FileResponse(tmp.name, media_type="image/png", filename="depth_map_gray.png")

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing image: {e}")


@app.get("/health")
async def health_check():
    gpu_count = torch.cuda.device_count()
    active = len([w for w in worker_pool if w.process and w.process.is_alive()])
    return {
        "status": "healthy" if active == len(worker_pool) else "degraded",
        "gpu_count": gpu_count,
        "active_workers": active,
        "total_workers": len(worker_pool),
        "model_id": DEFAULT_MODEL_ID,
    }

# Depth Anything 3 (REVPT integration)

FastAPI server wrapping [Depth Anything 3](https://github.com/ByteDance-Seed/depth-anything-3) for use as a REVPT tool. Drop-in replacement for the previous Depth Anything V2 service — same HTTP endpoint contract.

## Install

Depth Anything 3 ships as a pip-installable package; weights download from HuggingFace on first inference.

```bash
git clone https://github.com/ByteDance-Seed/depth-anything-3.git
cd depth-anything-3
pip install xformers "torch>=2" torchvision
pip install -e .
cd -

pip install -r tools/Depth-Anything-V3/requirements.txt
```

## Model selection

The default model is `depth-anything/DA3MONO-LARGE` (0.35B params, monocular relative depth — closest to V2's behavior). Override via env var to use a different variant:

| `DA3_MODEL_ID` | Output |
|---|---|
| `depth-anything/DA3MONO-LARGE` (default) | Relative depth, drop-in for V2 |
| `depth-anything/DA3METRIC-LARGE` | Metric depth (meters), via `metric_depth = focal * raw / 300` |

```bash
export DA3_MODEL_ID="depth-anything/DA3METRIC-LARGE"
```

## Serve

Via REVPT's launcher (recommended):

```bash
python tools/lanuch_tools.py --config tools/tools_config_2.json
```

Standalone:

```bash
cd tools/Depth-Anything-V3
uvicorn multi_deploy:app --host 0.0.0.0 --port 9991
```

## Endpoints

All endpoints accept a single image in the `file` form field.

| Method | Path | Returns |
|---|---|---|
| POST | `/predict/color_depth` | Colorized PNG (matplotlib `Spectral_r`) |
| POST | `/predict/depth` | Grayscale PNG |
| POST | `/predict/raw_depth_array` | JSON: `{depth_array, width, height, min_depth, max_depth}` |
| GET | `/health` | `{status, gpu_count, active_workers, total_workers, model_id}` |

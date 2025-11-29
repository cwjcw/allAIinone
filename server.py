import json
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import requests
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from openai import OpenAI
from wan_video_sdk_gui import generate_video_sdk
from wan_i2v_sdk import run_i2v_async_then_wait, I2VSDKError
from wan_kf2v_sdk import run_kf2v_async_then_wait, KF2VSDKError

load_dotenv()

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
DASHSCOPE_API_KEY = os.getenv("DASHSCOPE_API_KEY")

MODEL_CACHE = Path("openrouter_models.json")
MEDIA_DIR = Path("media")
MEDIA_DIR.mkdir(exist_ok=True)
DEFAULT_AUDIO_URL = "/media/admin.mp3"


def ensure_keys() -> None:
    if not OPENROUTER_API_KEY:
        raise RuntimeError("请在 .env 中配置 OPENROUTER_API_KEY")
    if not DASHSCOPE_API_KEY:
        raise RuntimeError("请在 .env 中配置 DASHSCOPE_API_KEY")


def is_free_model(pricing: Optional[Dict[str, Any]], model_id: str = "") -> bool:
    """
    判断免费：有定价则要求所有数值<=0；若无定价，针对 deepseek 系列默认视为免费。
    """
    if pricing:
        costs = []
        for v in pricing.values():
            try:
                costs.append(float(v))
            except Exception:
                continue
        if costs:
            return max(costs) <= 0
    # 没有定价信息，针对 deepseek 系列默认标记为免费
    if model_id.lower().startswith("deepseek"):
        return True
    return False


def supports_image(model_id: str) -> bool:
    lower = model_id.lower()
    keywords = ["image", "dalle", "sd", "flux", "gpt-image", "stable", "vision"]
    return any(k in lower for k in keywords)


def fetch_openrouter_models() -> List[Dict[str, Any]]:
    ensure_keys()
    headers = {"Authorization": f"Bearer {OPENROUTER_API_KEY}"}
    resp = requests.get(
        "https://openrouter.ai/api/v1/models", headers=headers, timeout=60
    )
    if not resp.ok:
        raise HTTPException(
            status_code=resp.status_code,
            detail=f"获取模型失败: {resp.text[:500]}",
        )
    raw = resp.json().get("data", [])
    enriched = []
    for m in raw:
        m = dict(m)
        m["free"] = is_free_model(m.get("pricing"), m.get("id", ""))
        m["supports_image"] = supports_image(m.get("id", ""))
        enriched.append(m)
    MODEL_CACHE.write_text(json.dumps(enriched, ensure_ascii=False, indent=2), encoding="utf-8")
    return enriched


def load_cached_models() -> List[Dict[str, Any]]:
    if MODEL_CACHE.exists():
        try:
            return json.loads(MODEL_CACHE.read_text(encoding="utf-8"))
        except Exception:
            return []
    return []


def get_models() -> List[Dict[str, Any]]:
    cached = load_cached_models()
    if cached:
        return cached
    return fetch_openrouter_models()


class Message(BaseModel):
    role: str
    content: Any


class ChatRequest(BaseModel):
    model: str
    messages: List[Message]


class ImageRequest(BaseModel):
    model: str
    prompt: str
    size: str = "1024x1024"


class VideoRequest(BaseModel):
    model: str = "wan2.2-t2v-plus"
    prompt: str
    duration: int = 5
    resolution: str = "480p"
    image_url: Optional[str] = None  # for i2v
    video_url: Optional[str] = None  # for edit
    audio_url: Optional[str] = None  # optional for i2v
    audio: Optional[bool] = None  # optional for i2v
    template: Optional[str] = None  # optional for i2v


class KF2VRequest(BaseModel):
    model: str = "wan2.2-kf2v-flash"
    prompt: str
    first_frame_url: str
    last_frame_url: str
    resolution: str = "720p"


def call_openrouter_chat(req: ChatRequest) -> Dict[str, Any]:
    ensure_keys()
    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": req.model,
        "messages": [msg.model_dump() for msg in req.messages],
    }
    resp = requests.post(
        "https://openrouter.ai/api/v1/chat/completions",
        headers=headers,
        json=payload,
        timeout=120,
    )
    if not resp.ok:
        raise HTTPException(status_code=resp.status_code, detail=resp.text[:500])
    return resp.json()


def call_openrouter_image(req: ImageRequest) -> Dict[str, Any]:
    ensure_keys()
    client = OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=OPENROUTER_API_KEY,
        default_headers={
            "HTTP-Referer": "https://localhost",
            "X-Title": "allAIinone",
            "User-Agent": "allAIinone/1.0",
        },
    )
    try:
        resp = client.images.generate(
            model=req.model,
            prompt=req.prompt,
            size=req.size,
            response_format="b64_json",
        )
        # openai SDK 返回对象，转 dict
        return {"data": [{"b64_json": resp.data[0].b64_json}]}
    except Exception as exc:
        # 捕获 SDK 内部异常，向前端返回可读信息
        raise HTTPException(status_code=502, detail=str(exc))


def call_dashscope_video(req: VideoRequest, mode: str) -> str:
    """
    文生视频走 dashscope SDK（同 wan_video_sdk_gui.py 的思路），i2v/edit 继续用 HTTP。
    """
    ensure_keys()

    def resolution_to_size(res: str) -> str:
        res = res.lower()
        if res == "480p":
            return "832*480"
        if res == "720p":
            return "1280*720"
        if res == "1080p":
            return "1440*810"  # SDK 示例里的 16:9 选项
        return "832*480"

    # 文生视频：直接复用 GUI 脚本的 SDK 封装，默认参数与 wan_video_sdk_gui.py 保持一致
    if mode == "t2v":
        try:
            return generate_video_sdk(
                prompt=req.prompt,
                model=req.model,
                size=resolution_to_size(req.resolution),
                duration=req.duration,
                audio_url=req.audio_url or DEFAULT_AUDIO_URL,
                api_key=DASHSCOPE_API_KEY,
            )
        except Exception as exc:
            raise HTTPException(status_code=502, detail=str(exc))

    # 图生视频：使用 SDK 异步调用 + wait
    if mode == "i2v":
        try:
            url = run_i2v_async_then_wait(
                api_key=DASHSCOPE_API_KEY,
                model=req.model,
                prompt=req.prompt,
                img_url=req.image_url or "",
                resolution=req.resolution,
                duration=req.duration,
                audio_url=req.audio_url or DEFAULT_AUDIO_URL,
                audio=req.audio,
                prompt_extend=True,
                watermark=None,
                seed=12345,
            )
            return url
        except I2VSDKError as exc:
            raise HTTPException(status_code=502, detail=str(exc))
        except Exception as exc:
            raise HTTPException(status_code=502, detail=str(exc))

app = FastAPI(title="AllAIInOne API")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/api/models")
def api_models() -> Dict[str, Any]:
    return {"data": get_models()}


@app.post("/api/models/refresh")
def api_models_refresh() -> Dict[str, Any]:
    return {"data": fetch_openrouter_models()}


@app.post("/api/chat")
def api_chat(req: ChatRequest) -> Dict[str, Any]:
    return call_openrouter_chat(req)


@app.post("/api/image")
def api_image(req: ImageRequest) -> Dict[str, Any]:
    return call_openrouter_image(req)


@app.post("/api/video/t2v")
def api_video_t2v(req: VideoRequest) -> Dict[str, Any]:
    url = call_dashscope_video(req, mode="t2v")
    return {"url": url}


@app.post("/api/video/i2v")
def api_video_i2v(req: VideoRequest) -> Dict[str, Any]:
    url = call_dashscope_video(req, mode="i2v")
    return {"url": url}


@app.post("/api/video/kf2v")
def api_video_kf2v(req: KF2VRequest) -> Dict[str, Any]:
    try:
        url = run_kf2v_async_then_wait(
            api_key=DASHSCOPE_API_KEY,
            model=req.model,
            prompt=req.prompt,
            first_frame_url=req.first_frame_url,
            last_frame_url=req.last_frame_url,
            resolution=req.resolution,
            prompt_extend=True,
            watermark=None,
            seed=12345,
        )
        return {"url": url}
    except KF2VSDKError as exc:
        raise HTTPException(status_code=502, detail=str(exc))


MAX_AUDIO_SIZE = 3 * 1024 * 1024
ALLOWED_AUDIO_SUFFIX = {".mp3", ".wav", ".m4a", ".aac"}


@app.post("/api/upload/audio")
async def api_upload_audio(file: UploadFile = File(...)) -> Dict[str, Any]:
    content = await file.read()
    if not content:
        raise HTTPException(status_code=400, detail="未收到音频文件")
    if len(content) > MAX_AUDIO_SIZE:
        raise HTTPException(status_code=413, detail="音频超过3M限制")
    suffix = Path(file.filename or "").suffix.lower()
    if suffix not in ALLOWED_AUDIO_SUFFIX:
        suffix = ".mp3"
    out_path = MEDIA_DIR / f"audio_{int(time.time())}{suffix}"
    out_path.write_bytes(content)
    return {"url": f"/media/{out_path.name}"}


@app.get("/")
def root() -> FileResponse:
    index_path = Path("public/index.html")
    if index_path.exists():
        return FileResponse(index_path)
    return FileResponse("README.md")


app.mount("/web", StaticFiles(directory="public", html=True), name="web")
app.mount("/media", StaticFiles(directory=MEDIA_DIR, html=False), name="media")

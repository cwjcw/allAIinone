import base64
import hashlib
import hmac
import json
import os
import secrets
import threading
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import requests
from dashscope import VideoSynthesis
from dotenv import load_dotenv
from fastapi import FastAPI, File, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from openai import OpenAI
from pydantic import BaseModel
from wan_i2v_sdk import I2VSDKError, run_i2v_async_then_wait
from wan_kf2v_sdk import KF2VSDKError, run_kf2v_async_then_wait

load_dotenv()

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
DASHSCOPE_API_KEY = os.getenv("DASHSCOPE_API_KEY")
SECRET_KEY = os.getenv("SECRET_KEY", "changeme-secret-key")

MODEL_CACHE = Path("openrouter_models.json")
MEDIA_DIR = Path("media")
MEDIA_DIR.mkdir(exist_ok=True)
AVATAR_DIR = MEDIA_DIR / "avatars"
AVATAR_DIR.mkdir(exist_ok=True)
DATA_DIR = Path("data")
DATA_DIR.mkdir(exist_ok=True)
USERS_FILE = DATA_DIR / "users.json"
USER_LOCK = threading.Lock()
CHAT_HISTORY_FILE = DATA_DIR / "chat_history.json"
CHAT_HISTORY_LOCK = threading.Lock()

MAX_AUDIO_SIZE = 3 * 1024 * 1024
ALLOWED_AUDIO_SUFFIX = {".mp3", ".wav", ".m4a", ".aac"}
MAX_IMAGE_SIZE = 10 * 1024 * 1024
ALLOWED_IMAGE_SUFFIX = {".jpg", ".jpeg", ".png", ".webp"}
I2V_DEFAULT_AUDIO = "https://help-static-aliyun-doc.aliyuncs.com/file-manage-files/zh-CN/20250925/ozwpvi/rap.mp3"


# --------------------------- 用户 & 权限 --------------------------- #
def hash_password(password: str, salt: Optional[str] = None) -> str:
    salt_bytes = salt.encode() if salt else secrets.token_bytes(16)
    if isinstance(salt_bytes, str):
        salt_bytes = salt_bytes.encode()
    dk = hashlib.pbkdf2_hmac("sha256", password.encode(), salt_bytes, 24000)
    return base64.urlsafe_b64encode(salt_bytes).decode() + "." + base64.urlsafe_b64encode(dk).decode()


def verify_password(password: str, stored: str) -> bool:
    try:
        salt_b64, hash_b64 = stored.split(".")
        salt = base64.urlsafe_b64decode(salt_b64.encode())
        expected = base64.urlsafe_b64decode(hash_b64.encode())
    except Exception:
        return False
    dk = hashlib.pbkdf2_hmac("sha256", password.encode(), salt, 24000)
    return hmac.compare_digest(dk, expected)


def load_users() -> Dict[str, Dict[str, Any]]:
    if not USERS_FILE.exists():
        return {}
    try:
        return json.loads(USERS_FILE.read_text(encoding="utf-8"))
    except Exception:
        return {}


def save_users(users: Dict[str, Dict[str, Any]]) -> None:
    USERS_FILE.write_text(json.dumps(users, ensure_ascii=False, indent=2), encoding="utf-8")


def load_history() -> Dict[str, List[Dict[str, Any]]]:
    if not CHAT_HISTORY_FILE.exists():
        return {}
    try:
        return json.loads(CHAT_HISTORY_FILE.read_text(encoding="utf-8"))
    except Exception:
        return {}


def save_history(data: Dict[str, List[Dict[str, Any]]]) -> None:
    CHAT_HISTORY_FILE.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def trim_old_history(records: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    cutoff = time.time() - 3 * 24 * 3600
    return [r for r in records if r.get("ts", 0) >= cutoff]


def append_history(username: str, entries: List[Dict[str, Any]]) -> None:
    with CHAT_HISTORY_LOCK:
        data = load_history()
        user_records = data.get(username, [])
        user_records = trim_old_history(user_records)
        user_records.extend(entries)
        data[username] = user_records
        save_history(data)


def get_recent_history(username: str) -> List[Dict[str, Any]]:
    with CHAT_HISTORY_LOCK:
        data = load_history()
        records = trim_old_history(data.get(username, []))
        data[username] = records
        save_history(data)
        return records


def promote_if_no_admin(users: Dict[str, Dict[str, Any]], username: str) -> None:
    """如果系统里没有管理员，则将当前用户设为管理员。"""
    has_admin = any(u.get("role") == "admin" for u in users.values())
    if not has_admin and username in users:
        users[username]["role"] = "admin"


def create_token(username: str, role: str, ttl_hours: int = 24) -> str:
    payload = {
        "username": username,
        "role": role,
        "exp": int(time.time()) + ttl_hours * 3600,
    }
    payload_bytes = json.dumps(payload, separators=(",", ":")).encode()
    sig = hmac.new(SECRET_KEY.encode(), payload_bytes, hashlib.sha256).digest()
    return (
        base64.urlsafe_b64encode(payload_bytes).decode().rstrip("=")
        + "."
        + base64.urlsafe_b64encode(sig).decode().rstrip("=")
    )


def verify_token(token: str) -> Dict[str, Any]:
    try:
        payload_b64, sig_b64 = token.split(".")
        payload_bytes = base64.urlsafe_b64decode(payload_b64 + "==")
        sig = base64.urlsafe_b64decode(sig_b64 + "==")
        expected = hmac.new(SECRET_KEY.encode(), payload_bytes, hashlib.sha256).digest()
        if not hmac.compare_digest(sig, expected):
            raise ValueError("bad signature")
        data = json.loads(payload_bytes.decode())
        if data.get("exp", 0) < time.time():
            raise ValueError("expired")
        return data
    except Exception:
        raise HTTPException(status_code=401, detail="登录已失效，请重新登录")


def ensure_keys() -> None:
    if not OPENROUTER_API_KEY:
        raise RuntimeError("请在 .env 中配置 OPENROUTER_API_KEY")
    if not DASHSCOPE_API_KEY:
        raise RuntimeError("请在 .env 中配置 DASHSCOPE_API_KEY")


def get_user_from_request(request: Request) -> Dict[str, Any]:
    auth = request.headers.get("Authorization", "")
    if not auth.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="请先登录")
    token = auth.split(" ", 1)[1]
    data = verify_token(token)
    username = data["username"]
    with USER_LOCK:
        users = load_users()
        user = users.get(username)
        if not user:
            raise HTTPException(status_code=401, detail="用户不存在")
    return user


def ensure_access(user: Dict[str, Any], feature: str) -> float:
    """
    feature: chat | image | video | upload
    return: 应扣费金额（仅付费用户且功能计费时返回）
    """
    role = user.get("role", "free")
    if role in ("vip", "admin"):
        return 0.0
    if role == "free":
        if feature != "chat":
            raise HTTPException(status_code=403, detail="免费用户仅支持对话功能")
        return 0.0
    if role == "paid":
        cost_map = {"image": 1.0, "video": 3.0}
        cost = cost_map.get(feature, 0.0)
        if cost > 0 and float(user.get("balance", 0)) < cost:
            raise HTTPException(status_code=402, detail="余额不足，请充值")
        return cost
    raise HTTPException(status_code=403, detail="无效角色")


def deduct_balance(username: str, amount: float) -> None:
    if amount <= 0:
        return
    with USER_LOCK:
        users = load_users()
        user = users.get(username)
        if not user:
            raise HTTPException(status_code=401, detail="用户不存在")
        user["balance"] = round(float(user.get("balance", 0)) - amount, 2)
        users[username] = user
        save_users(users)


# --------------------------- 模型与请求 --------------------------- #
def is_free_model(pricing: Optional[Dict[str, Any]], model_id: str = "") -> bool:
    if pricing:
        costs = []
        for v in pricing.values():
            try:
                costs.append(float(v))
            except Exception:
                continue
        if costs:
            return max(costs) <= 0
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
    resp = requests.get("https://openrouter.ai/api/v1/models", headers=headers, timeout=60)
    if not resp.ok:
        raise HTTPException(status_code=resp.status_code, detail=f"获取模型失败: {resp.text}")
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
    image_url: Optional[str] = None  # 兼容单个
    image_url_list: Optional[List[str]] = None  # 最多 4 个 URL
    image_b64: Optional[str] = None  # 兼容单个 base64
    image_b64_list: Optional[List[str]] = None  # 最多 4 个 base64


class VideoRequest(BaseModel):
    model: str = "wan2.2-t2v-plus"
    prompt: str
    duration: int = 5
    resolution: str = "480p"
    image_url: Optional[str] = None  # for i2v
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
    payload = {"model": req.model, "messages": [msg.model_dump() for msg in req.messages]}
    resp = requests.post(
        "https://openrouter.ai/api/v1/chat/completions",
        headers=headers,
        json=payload,
        timeout=120,
    )
    if not resp.ok:
        raise HTTPException(status_code=resp.status_code, detail=resp.text)
    return resp.json()


def call_openrouter_image(req: ImageRequest) -> Dict[str, Any]:
    ensure_keys()
    lower_model = req.model.lower()
    if "gemini" in lower_model:
        headers = {
            "Authorization": f"Bearer {OPENROUTER_API_KEY}",
            "Content-Type": "application/json",
            "HTTP-Referer": os.getenv("OPENROUTER_SITE_URL", "http://localhost"),
            "X-Title": os.getenv("OPENROUTER_APP_TITLE", "allAIinone"),
            "User-Agent": "allAIinone/1.0",
        }
        content_parts: List[Dict[str, Any]] = []
        url_list = []
        if req.image_url:
            url_list.append(req.image_url)
        if req.image_url_list:
            url_list.extend([u for u in req.image_url_list if u])
        for url in url_list[:4]:
            content_parts.append({"type": "image_url", "image_url": {"url": url}})
        b64_list = []
        if req.image_b64_list:
            b64_list.extend(req.image_b64_list[:4])
        if req.image_b64:
            b64_list.append(req.image_b64)
        for b64 in b64_list[:4]:
            content_parts.append({"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}})
        if req.prompt:
            content_parts.append({"type": "text", "text": req.prompt})
        if not content_parts:
            raise HTTPException(status_code=400, detail="请输入提示词或图片")

        payload = {
            "model": req.model,
            "messages": [{"role": "user", "content": content_parts}],
            "modalities": ["image", "text"],
        }
        resp = None
        for attempt in range(3):
            resp = requests.post(
                "https://openrouter.ai/api/v1/chat/completions",
                headers=headers,
                json=payload,
                timeout=120,
            )
            if resp.status_code < 500:
                break
            if attempt < 2:
                time.sleep(1 + attempt)
        if resp is None:
            raise HTTPException(status_code=502, detail="请求未发送成功")
        if not resp.ok:
            raise HTTPException(status_code=resp.status_code, detail=resp.text)
        data = resp.json()
        try:
            message = data["choices"][0]["message"]
            images = message.get("images") or []
            if not images and isinstance(message.get("content"), str):
                # 有些模型可能把 data URL 放在 content
                content = message["content"]
                if "base64," in content:
                    images = [{"image_url": {"url": content}}]
            if not images:
                reasoning = message.get("reasoning") or ""
                snippet = reasoning[:300] if isinstance(reasoning, str) else str(message)[:300]
                raise HTTPException(
                    status_code=400,
                    detail=f"模型未返回图片，可能生成失败或被安全策略拦截。提示: {snippet}",
                )
            b64_url = images[0]["image_url"]["url"]
            prefix = "base64,"
            pos = b64_url.find(prefix)
            if pos == -1:
                raise ValueError("unexpected image_url format")
            b64 = b64_url[pos + len(prefix):]
            return {"data": [{"b64_json": b64}]}
        except HTTPException:
            raise
        except Exception as exc:
            detail = ""
            try:
                detail = f" 响应片段: {str(data)[:400]}"
            except Exception:
                pass
            raise HTTPException(status_code=500, detail=f"解析图片失败: {exc}{detail}")

    client = OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=OPENROUTER_API_KEY,
        default_headers={
            "HTTP-Referer": os.getenv("OPENROUTER_SITE_URL", "http://localhost"),
            "X-Title": os.getenv("OPENROUTER_APP_TITLE", "allAIinone"),
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
        if not resp.data:
            raise ValueError("未返回 data 字段")
        return {"data": [{"b64_json": resp.data[0].b64_json}]}
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"生成失败: {exc}")


def call_dashscope_video(req: VideoRequest, mode: str) -> str:
    ensure_keys()

    def resolution_to_size(res: str) -> str:
        res = res.lower()
        if res == "480p":
            return "832*480"
        if res == "720p":
            return "1280*720"
        if res == "1080p":
            return "1440*810"
        return "832*480"

    if mode == "t2v":
        try:
            rsp = VideoSynthesis.call(
                api_key=DASHSCOPE_API_KEY,
                model=req.model,
                prompt=req.prompt,
                size=resolution_to_size(req.resolution),
                duration=req.duration,
                negative_prompt="",
                prompt_extend=True,
                watermark=None,
                seed=12345,
            )
            if rsp.status_code == 200:
                url = getattr(rsp.output, "video_url", None)
                if not url:
                    raise HTTPException(status_code=500, detail="未返回视频地址，请检查模型权限或参数")
                return url
            raise HTTPException(
                status_code=rsp.status_code,
                detail=f"dashscope error: code={rsp.code}, message={rsp.message}",
            )
        except Exception as exc:
            raise HTTPException(status_code=502, detail=str(exc))

    if mode == "i2v":
        try:
            url = run_i2v_async_then_wait(
                api_key=DASHSCOPE_API_KEY,
                model=req.model,
                prompt=req.prompt,
                img_url=req.image_url or "",
                resolution=req.resolution,
                duration=req.duration,
                audio_url=req.audio_url or I2V_DEFAULT_AUDIO,
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

    raise HTTPException(status_code=400, detail=f"不支持的模式: {mode}")


# --------------------------- FastAPI --------------------------- #
app = FastAPI(title="AllAIInOne API")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Auth & 用户管理
class AuthPayload(BaseModel):
    username: str
    password: str


class AdminUpdatePayload(BaseModel):
    username: str
    role: Optional[str] = None
    balance: Optional[float] = None
    avatar: Optional[str] = None


@app.post("/api/register")
def api_register(payload: AuthPayload) -> Dict[str, Any]:
    username = payload.username.strip()
    password = payload.password.strip()
    if not username or not password:
        raise HTTPException(status_code=400, detail="用户名和密码不能为空")
    with USER_LOCK:
        users = load_users()
        if username in users:
            raise HTTPException(status_code=400, detail="用户已存在")
        role = "admin" if not users else "free"
        pwd_hash = hash_password(password)
        users[username] = {
            "username": username,
            "password_hash": pwd_hash,
            "role": role,
            "balance": 0.0,
            "avatar": "",
        }
        save_users(users)
    token = create_token(username, role)
    return {"token": token, "role": role}


@app.post("/api/login")
def api_login(payload: AuthPayload) -> Dict[str, Any]:
    username = payload.username.strip()
    password = payload.password.strip()
    with USER_LOCK:
        users = load_users()
        user = users.get(username)
        if not user or not verify_password(password, user.get("password_hash", "")):
            raise HTTPException(status_code=401, detail="用户名或密码错误")
        promote_if_no_admin(users, username)
        role = users[username].get("role", "free")
        save_users(users)
    token = create_token(username, role)
    return {"token": token, "role": role}


@app.get("/api/me")
def api_me(request: Request) -> Dict[str, Any]:
    user = get_user_from_request(request)
    data = {k: v for k, v in user.items() if k != "password_hash"}
    return {"user": data}


@app.get("/api/admin/users")
def api_admin_users(request: Request) -> Dict[str, Any]:
    user = get_user_from_request(request)
    if user.get("role") != "admin":
        raise HTTPException(status_code=403, detail="仅管理员可查看用户列表")
    with USER_LOCK:
        users = load_users()
    cleaned = []
    for u in users.values():
        cleaned.append({k: v for k, v in u.items() if k != "password_hash"})
    return {"users": cleaned}


@app.post("/api/admin/users/update")
def api_admin_update(request: Request, payload: AdminUpdatePayload) -> Dict[str, Any]:
    user = get_user_from_request(request)
    if user.get("role") != "admin":
        raise HTTPException(status_code=403, detail="仅管理员可操作")
    with USER_LOCK:
        users = load_users()
        target = users.get(payload.username)
        if not target:
            raise HTTPException(status_code=404, detail="目标用户不存在")
        if payload.role:
            target["role"] = payload.role
        if payload.balance is not None:
            target["balance"] = float(payload.balance)
        if payload.avatar is not None:
            target["avatar"] = payload.avatar
        users[payload.username] = target
        save_users(users)
    return {"message": "updated"}


# 模型
@app.get("/api/models")
def api_models(request: Request) -> Dict[str, Any]:
    _ = get_user_from_request(request)
    return {"data": get_models()}


@app.post("/api/models/refresh")
def api_models_refresh(request: Request) -> Dict[str, Any]:
    user = get_user_from_request(request)
    if user.get("role") not in ("admin", "vip"):
        raise HTTPException(status_code=403, detail="仅管理员或 VIP 可刷新模型")
    return {"data": fetch_openrouter_models()}


@app.post("/api/chat")
def api_chat(req: ChatRequest, request: Request) -> Dict[str, Any]:
    user = get_user_from_request(request)
    ensure_access(user, "chat")
    result = call_openrouter_chat(req)
    try:
        user_msg = ""
        if req.messages:
            user_msg = str(req.messages[-1].content)
        assistant_msg = ""
        if result.get("choices"):
            assistant_msg = result["choices"][0]["message"].get("content", "")
        ts = time.time()
        entries = []
        if user_msg:
            entries.append({"role": "user", "content": user_msg, "ts": ts})
        if assistant_msg:
            entries.append({"role": "assistant", "content": assistant_msg, "ts": ts})
        if entries:
            append_history(user["username"], entries)
    except Exception:
        pass
    return result


@app.get("/api/chat/history")
def api_chat_history(request: Request) -> Dict[str, Any]:
    user = get_user_from_request(request)
    history = get_recent_history(user["username"])
    return {"history": history, "notice": "已显示最近3天内的对话记录"}


@app.post("/api/image")
def api_image(req: ImageRequest, request: Request) -> Dict[str, Any]:
    user = get_user_from_request(request)
    cost = ensure_access(user, "image")
    result = call_openrouter_image(req)
    if cost > 0:
        deduct_balance(user["username"], cost)
    return result


@app.post("/api/video/t2v")
def api_video_t2v(req: VideoRequest, request: Request) -> Dict[str, Any]:
    user = get_user_from_request(request)
    cost = ensure_access(user, "video")
    url = call_dashscope_video(req, mode="t2v")
    if cost > 0:
        deduct_balance(user["username"], cost)
    return {"url": url}


@app.post("/api/video/i2v")
def api_video_i2v(req: VideoRequest, request: Request) -> Dict[str, Any]:
    user = get_user_from_request(request)
    cost = ensure_access(user, "video")
    url = call_dashscope_video(req, mode="i2v")
    if cost > 0:
        deduct_balance(user["username"], cost)
    return {"url": url}


@app.post("/api/video/kf2v")
def api_video_kf2v(req: KF2VRequest, request: Request) -> Dict[str, Any]:
    user = get_user_from_request(request)
    cost = ensure_access(user, "video")
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
        if cost > 0:
            deduct_balance(user["username"], cost)
        return {"url": url}
    except KF2VSDKError as exc:
        raise HTTPException(status_code=502, detail=str(exc))


@app.post("/api/upload/audio")
async def api_upload_audio(request: Request, file: UploadFile = File(...)) -> Dict[str, Any]:
    user = get_user_from_request(request)
    ensure_access(user, "chat")  # 上传仅需登录即可
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


@app.post("/api/upload/image")
async def api_upload_image(request: Request, file: UploadFile = File(...)) -> Dict[str, Any]:
    user = get_user_from_request(request)
    ensure_access(user, "chat")  # 上传仅需登录即可
    content = await file.read()
    if not content:
        raise HTTPException(status_code=400, detail="未收到图片文件")
    if len(content) > MAX_IMAGE_SIZE:
        raise HTTPException(status_code=413, detail="图片超过10M限制")
    suffix = Path(file.filename or "").suffix.lower()
    if suffix not in ALLOWED_IMAGE_SUFFIX:
        raise HTTPException(status_code=400, detail="仅支持 jpg/jpeg/png/webp")
    out_path = MEDIA_DIR / f"image_{int(time.time())}{suffix}"
    out_path.write_bytes(content)
    return {"url": f"/media/{out_path.name}"}


@app.post("/api/me/avatar")
async def api_me_avatar(request: Request, file: UploadFile = File(...)) -> Dict[str, Any]:
    user = get_user_from_request(request)
    content = await file.read()
    if not content:
        raise HTTPException(status_code=400, detail="未收到头像文件")
    if len(content) > MAX_IMAGE_SIZE:
        raise HTTPException(status_code=413, detail="图片超过10M限制")
    suffix = Path(file.filename or "").suffix.lower()
    if suffix not in ALLOWED_IMAGE_SUFFIX:
        suffix = ".png"
    filename = f"avatar_{user['username']}_{int(time.time())}{suffix}"
    out_path = AVATAR_DIR / filename
    out_path.write_bytes(content)
    with USER_LOCK:
        users = load_users()
        if user["username"] in users:
            users[user["username"]]["avatar"] = f"/media/avatars/{filename}"
            save_users(users)
    return {"avatar": f"/media/avatars/{filename}"}


@app.get("/")
def root() -> FileResponse:
    index_path = Path("public/index.html")
    if index_path.exists():
        return FileResponse(index_path)
    return FileResponse("README.md")


app.mount("/web", StaticFiles(directory="public", html=True), name="web")
app.mount("/media", StaticFiles(directory=MEDIA_DIR, html=False), name="media")

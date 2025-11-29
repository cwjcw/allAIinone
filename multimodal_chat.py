import argparse
import base64
import mimetypes
import os
from pathlib import Path
from typing import List

import requests
from dotenv import load_dotenv


def load_api_key() -> str:
    load_dotenv()
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        raise RuntimeError("请在 .env 或环境变量中设置 OPENROUTER_API_KEY")
    return api_key


def encode_image_to_data_url(path: Path) -> str:
    mime, _ = mimetypes.guess_type(path)
    mime = mime or "image/png"
    b64 = base64.b64encode(path.read_bytes()).decode("utf-8")
    return f"data:{mime};base64,{b64}"


def build_messages(prompt: str, image_urls: List[str]) -> List[dict]:
    """
    构建 chat/completions 的 messages，按文档推荐先放文本，再放图片。
    image_urls 可以是 http/https URL，也可以是 data URL。
    """
    content = [{"type": "text", "text": prompt}]
    for url in image_urls:
        content.append({"type": "image_url", "image_url": {"url": url}})
    return [{"role": "user", "content": content}]


def call_chat(prompt: str, image_urls: List[str], model: str) -> dict:
    api_key = load_api_key()
    url = "https://openrouter.ai/api/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "Accept": "application/json",
        # 统计头，可选
        "HTTP-Referer": "https://localhost",
        "X-Title": "allAIinone-multimodal-demo",
    }
    payload = {
        "model": model,
        "messages": build_messages(prompt, image_urls),
    }
    resp = requests.post(url, headers=headers, json=payload, timeout=120)
    if not resp.ok:
        raise RuntimeError(
            f"请求失败 status={resp.status_code}, content_type={resp.headers.get('content-type')}, "
            f"body={resp.text[:500]}"
        )
    return resp.json()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="OpenRouter 多模态示例：文本-only / 远程图片URL / 本地图片 data URL"
    )
    parser.add_argument("--prompt", required=True, help="文本提示词")
    parser.add_argument(
        "--image-url",
        action="append",
        default=[],
        help="要发送的远程图片 URL，可重复",
    )
    parser.add_argument(
        "--image-path",
        action="append",
        default=[],
        help="要发送的本地图片路径，可重复",
    )
    parser.add_argument(
        "--model",
        default="google/gemini-2.0-flash-001",
        help="模型名称，默认 google/gemini-2.0-flash-001",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    data_urls: List[str] = []
    for p in args.image_path:
        path = Path(p)
        if not path.exists():
            raise FileNotFoundError(f"本地图片不存在: {path}")
        data_urls.append(encode_image_to_data_url(path))

    # 保证顺序：文本 -> 远程URL -> 本地图片(data URL)
    image_inputs = list(args.image_url) + data_urls

    result = call_chat(args.prompt, image_inputs, args.model)
    if result.get("choices"):
        msg = result["choices"][0]["message"]
        text = msg.get("content") or msg.get("text") or msg
        print("模型回复:")
        print(text)
    else:
        print("No choices returned in the response.")


if __name__ == "__main__":
    main()

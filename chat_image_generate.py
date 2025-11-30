import base64
import os
import pathlib
import time
from typing import Dict, List

import requests

try:
    # Optional: allow running even if python-dotenv is not installed
    from dotenv import load_dotenv  # type: ignore

    load_dotenv()
except Exception:
    pass

API_KEY_REF = os.getenv("OPENROUTER_API_KEY")
if not API_KEY_REF:
    raise RuntimeError("请在 .env 或环境变量中设置 OPENROUTER_API_KEY")

url = "https://openrouter.ai/api/v1/chat/completions"
headers: Dict[str, str] = {
    "Authorization": f"Bearer {API_KEY_REF}",
    "Content-Type": "application/json",
    # OpenRouter 推荐带上来源信息，部分反代环境下可减少 502 概率
    "HTTP-Referer": os.getenv("OPENROUTER_SITE_URL", "http://localhost"),
    "X-Title": os.getenv("OPENROUTER_APP_TITLE", "allAIinone-image-demo"),
}

payload = {
    "model": "google/gemini-3-pro-image-preview",
    "messages": [
        {
            "role": "user",
            "content": "Generate a beautiful sunset over mountains",
        }
    ],
    "modalities": ["image", "text"],
}

# 简单的 5xx 重试，减少偶发 502
response = None
for attempt in range(3):
    response = requests.post(url, headers=headers, json=payload, timeout=120)
    if response.status_code < 500:
        break
    if attempt < 2:
        time.sleep(1 + attempt)
if response is None:
    raise RuntimeError("请求未发送成功。")

response.raise_for_status()
result = response.json()

choices: List[dict] = result.get("choices", [])
if not choices:
    raise RuntimeError("No choices returned in the response.")

message = choices[0]["message"]
images = message.get("images")
if not images:
    raise RuntimeError("No images returned in the response.")

out_dir = pathlib.Path("generated_images")
out_dir.mkdir(exist_ok=True)

for idx, image in enumerate(images, start=1):
    image_url = image["image_url"]["url"]  # Base64 data URL
    prefix = "base64,"  # 去掉 data URL 前缀
    pos = image_url.find(prefix)
    if pos == -1:
        raise RuntimeError("Unexpected image_url format, no base64 prefix found.")
    b64_data = image_url[pos + len(prefix) :]
    img_bytes = base64.b64decode(b64_data)

    out_file = out_dir / f"image_{idx}.png"
    out_file.write_bytes(img_bytes)
    print(f"Saved: {out_file.resolve()}")

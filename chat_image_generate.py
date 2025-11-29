import base64
import os
import pathlib

import requests
from dotenv import load_dotenv

# 加载 .env 中的 OPENROUTER_API_KEY
load_dotenv()
API_KEY_REF = os.getenv("OPENROUTER_API_KEY")
if not API_KEY_REF:
    raise RuntimeError("请在 .env 或环境变量中设置 OPENROUTER_API_KEY")

url = "https://openrouter.ai/api/v1/chat/completions"
headers = {
    "Authorization": f"Bearer {API_KEY_REF}",
    "Content-Type": "application/json",
}

payload = {
    "model": "google/gemini-2.5-flash-image-preview",
    "messages": [
        {
            "role": "user",
            "content": "Generate a beautiful sunset over mountains",
        }
    ],
    "modalities": ["image", "text"],
}

response = requests.post(url, headers=headers, json=payload, timeout=120)
response.raise_for_status()
result = response.json()

# The generated image will be in the assistant message
if not result.get("choices"):
    raise RuntimeError("No choices returned in the response.")

message = result["choices"][0]["message"]
images = message.get("images")
if not images:
    raise RuntimeError("No images returned in the response.")

out_dir = pathlib.Path("generated_images")
out_dir.mkdir(exist_ok=True)

for idx, image in enumerate(images, start=1):
    image_url = image["image_url"]["url"]  # Base64 data URL
    # 去掉 data URL 前缀
    prefix = "base64,"
    pos = image_url.find(prefix)
    if pos == -1:
        raise RuntimeError("Unexpected image_url format, no base64 prefix found.")
    b64_data = image_url[pos + len(prefix) :]
    img_bytes = base64.b64decode(b64_data)

    out_file = out_dir / f"image_{idx}.png"
    out_file.write_bytes(img_bytes)
    print(f"Saved: {out_file.resolve()}")

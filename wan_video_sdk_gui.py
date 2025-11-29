import os
import threading
import time
from http import HTTPStatus
from pathlib import Path
import tkinter as tk
from tkinter import messagebox

import dashscope
import requests
from dashscope import VideoSynthesis
from dotenv import load_dotenv

# 默认参数与可选项
DEFAULT_MODEL = "wan2.2-t2v-plus"
MODEL_OPTIONS = [
    "wan2.2-t2v-plus",
    "wan2.5-t2v-preview",
    "wanx2.1-t2v-turbo",
    "wanx2.1-t2v-plus",
]

DEFAULT_SIZE = "832*480"
SIZE_OPTIONS = ["832*480", "1280*720", "1440*810"]

DEFAULT_DURATION = 5
DURATION_OPTIONS = [5, 10]

DEFAULT_PROMPT = (
    "一幅史诗级可爱的场景。一只小巧可爱的卡通小猫将军，身穿细节精致的金色盔甲，"
    "头戴一个稍大的头盔，勇敢地站在悬崖上。他骑着一匹虽小但英勇的战马，说："
    "“青海长云暗雪山，孤城遥望玉门关。黄沙百战穿金甲，不破楼兰终不还。”。"
    "悬崖下方，一支由老鼠组成的、数量庞大、无穷无尽的军队正带着临时制作的武器向前冲锋。"
    "这是一个戏剧性的、大规模的战斗场景，灵感来自中国古代的战争史诗。远处的雪山上空，天空乌云密布。"
    "整体氛围是“可爱”与“霸气”的搞笑和史诗般的融合。"
)

DEFAULT_AUDIO_URL = (
    "https://help-static-aliyun-doc.aliyuncs.com/file-manage-files/zh-CN/20250923/hbiayh/%E4%BB%8E%E5%86%9B%E8%A1%8C.mp3"
)

# 使用北京地域，如需新加坡请改为 https://dashscope-intl.aliyuncs.com/api/v1
dashscope.base_http_api_url = "https://dashscope.aliyuncs.com/api/v1"


def load_api_key() -> str:
    load_dotenv()
    api_key = os.getenv("DASHSCOPE_API_KEY")
    if not api_key:
        raise RuntimeError("请在 .env 中设置 DASHSCOPE_API_KEY")
    return api_key


def download_video(video_url: str) -> Path:
    out_dir = Path("generated_videos")
    out_dir.mkdir(exist_ok=True)

    suffix = ".mp4"
    for ext in [".mp4", ".mov", ".mkv", ".avi"]:
        if video_url.lower().split("?")[0].endswith(ext):
            suffix = ext
            break
    out_path = out_dir / f"video_{int(time.time())}{suffix}"

    resp = requests.get(video_url, stream=True, timeout=300)
    if not resp.ok:
        raise RuntimeError(f"下载视频失败 status={resp.status_code}, body={resp.text[:300]}")
    with out_path.open("wb") as f:
        for chunk in resp.iter_content(chunk_size=8192):
            if chunk:
                f.write(chunk)
    return out_path


def run_generation(
    api_key: str,
    model: str,
    prompt: str,
    size: str,
    duration: int,
    watermark: str,
    audio_url: str,
    status_var: tk.StringVar,
    btn: tk.Button,
) -> None:
    try:
        status_var.set("任务提交中，请稍候...")
        btn.config(state=tk.DISABLED)

        rsp = VideoSynthesis.call(
            api_key=api_key,
            model=model,
            prompt=prompt,
            size=size,
            duration=duration,
            negative_prompt="",
            watermark=watermark if watermark.strip() else None,
            prompt_extend=True,
            seed=12345,
            audio_url=audio_url.strip() or None,
            # audio=True,  # 若需要音频输出可启用
        )
        if rsp.status_code == HTTPStatus.OK:
            video_url = getattr(rsp.output, "video_url", None)
            if not video_url:
                raise RuntimeError("未返回视频地址")
            saved_path = download_video(video_url)
            status_var.set("生成成功")
            messagebox.showinfo("生成成功", f"视频已保存：\n{saved_path.resolve()}\n\n原始地址：\n{video_url}")
            print("video_url:", video_url)
            print("saved to:", saved_path)
        else:
            msg = f"Failed, status_code: {rsp.status_code}, code: {rsp.code}, message: {rsp.message}"
            status_var.set("生成失败")
            messagebox.showerror("生成失败", msg)
            print(msg)
    except Exception as exc:
        status_var.set("异常")
        messagebox.showerror("错误", str(exc))
        print(exc)
    finally:
        btn.config(state=tk.NORMAL)


def main() -> None:
    api_key = load_api_key()

    root = tk.Tk()
    root.title("Wan 视频生成（dashscope SDK）")

    # 模型
    tk.Label(root, text="模型:").grid(row=0, column=0, sticky="w", padx=5, pady=5)
    model_var = tk.StringVar(value=DEFAULT_MODEL)
    tk.OptionMenu(root, model_var, *MODEL_OPTIONS).grid(row=0, column=1, sticky="ew", padx=5, pady=5)

    # 提示词
    tk.Label(root, text="提示词:").grid(row=1, column=0, sticky="nw", padx=5, pady=5)
    prompt_text = tk.Text(root, width=70, height=6)
    prompt_text.insert("1.0", DEFAULT_PROMPT)
    prompt_text.grid(row=1, column=1, sticky="ew", padx=5, pady=5)

    # 尺寸
    tk.Label(root, text="尺寸:").grid(row=2, column=0, sticky="w", padx=5, pady=5)
    size_var = tk.StringVar(value=DEFAULT_SIZE)
    tk.OptionMenu(root, size_var, *SIZE_OPTIONS).grid(row=2, column=1, sticky="ew", padx=5, pady=5)

    # 时长
    tk.Label(root, text="时长(秒):").grid(row=3, column=0, sticky="w", padx=5, pady=5)
    duration_var = tk.StringVar(value=str(DEFAULT_DURATION))
    tk.OptionMenu(root, duration_var, *[str(x) for x in DURATION_OPTIONS]).grid(row=3, column=1, sticky="ew", padx=5, pady=5)

    # 水印
    tk.Label(root, text="水印文本(可空):").grid(row=4, column=0, sticky="w", padx=5, pady=5)
    watermark_var = tk.StringVar(value="")
    tk.Entry(root, textvariable=watermark_var, width=50).grid(row=4, column=1, sticky="ew", padx=5, pady=5)

    # 音频 URL
    tk.Label(root, text="音频URL(可空):").grid(row=5, column=0, sticky="w", padx=5, pady=5)
    audio_var = tk.StringVar(value=DEFAULT_AUDIO_URL)
    tk.Entry(root, textvariable=audio_var, width=50).grid(row=5, column=1, sticky="ew", padx=5, pady=5)

    status_var = tk.StringVar(value="等待提交")
    tk.Label(root, textvariable=status_var, fg="blue").grid(row=6, column=0, columnspan=2, sticky="w", padx=5, pady=5)

    def on_submit() -> None:
        try:
            model = model_var.get().strip()
            prompt = prompt_text.get("1.0", tk.END).strip()
            size = size_var.get().strip()
            duration = int(duration_var.get().strip())
            watermark = watermark_var.get()
            audio_url = audio_var.get()

            if model not in MODEL_OPTIONS:
                raise ValueError(f"模型需为 {MODEL_OPTIONS}")
            if size not in SIZE_OPTIONS:
                raise ValueError(f"尺寸需为 {SIZE_OPTIONS}")
            if duration not in DURATION_OPTIONS:
                raise ValueError(f"时长需为 {DURATION_OPTIONS}")
            if not prompt:
                raise ValueError("提示词不能为空")

            threading.Thread(
                target=run_generation,
                args=(api_key, model, prompt, size, duration, watermark, audio_url, status_var, submit_btn),
                daemon=True,
            ).start()
        except Exception as exc:
            messagebox.showerror("参数错误", str(exc))

    submit_btn = tk.Button(root, text="生成视频", command=on_submit)
    submit_btn.grid(row=7, column=0, columnspan=2, pady=10)

    root.grid_columnconfigure(1, weight=1)
    root.mainloop()


if __name__ == "__main__":
    main()

import tkinter as tk
from tkinter import filedialog, messagebox, simpledialog

from multimodal_chat import call_chat, encode_image_to_data_url

DEFAULT_MODEL = "google/gemini-2.0-flash-001"


def ask_model(root: tk.Tk) -> str:
    model = simpledialog.askstring(
        "选择模型",
        f"输入模型名称（回车使用默认）\n默认: {DEFAULT_MODEL}",
        parent=root,
    )
    return model.strip() if model else DEFAULT_MODEL


def ask_prompt(root: tk.Tk) -> str:
    prompt = simpledialog.askstring("提示词", "请输入文本提示词：", parent=root)
    if not prompt:
        raise SystemExit("已取消：未输入提示词")
    return prompt


def ask_mode(root: tk.Tk) -> str:
    mode = simpledialog.askstring(
        "选择模式",
        "选择输入方式：\n1. 仅文本\n2. 远程图片 URL\n3. 本地图片文件",
        initialvalue="1",
        parent=root,
    )
    if not mode:
        raise SystemExit("已取消：未选择模式")
    return mode.strip()


def collect_image_urls(root: tk.Tk) -> list[str]:
    urls = simpledialog.askstring(
        "图片 URL",
        "输入一个或多个图片 URL，逗号分隔：",
        parent=root,
    )
    if not urls:
        return []
    return [u.strip() for u in urls.split(",") if u.strip()]


def collect_image_paths(root: tk.Tk) -> list[str]:
    paths = filedialog.askopenfilenames(
        title="选择要发送的图片",
        filetypes=[("Images", "*.png *.jpg *.jpeg *.webp *.gif"), ("All files", "*.*")],
    )
    return list(paths)


def main() -> None:
    root = tk.Tk()
    root.withdraw()  # 隐藏主窗口，只用对话框

    try:
        prompt = ask_prompt(root)
        mode = ask_mode(root)
        model = ask_model(root)

        image_inputs: list[str] = []
        if mode == "2":
            image_inputs.extend(collect_image_urls(root))
        elif mode == "3":
            paths = collect_image_paths(root)
            for p in paths:
                image_inputs.append(encode_image_to_data_url(Path(p)))
        # mode == "1" 为纯文本，不添加图片

        result = call_chat(prompt, image_inputs, model)
        if result.get("choices"):
            msg = result["choices"][0]["message"]
            text = msg.get("content") or msg.get("text") or str(msg)
            messagebox.showinfo("模型回复", text)
        else:
            messagebox.showwarning("提示", "响应中没有 choices。")
    except Exception as exc:
        messagebox.showerror("错误", str(exc))
    finally:
        root.destroy()


if __name__ == "__main__":
    from pathlib import Path

    main()

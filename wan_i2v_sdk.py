"""
阿里万相图生视频：使用 DashScope SDK 的 async_call + wait 封装。
"""
import dashscope
from dashscope import VideoSynthesis
from http import HTTPStatus


# 北京地域；如需新加坡请改为 intl 域名
dashscope.base_http_api_url = "https://dashscope.aliyuncs.com/api/v1"


class I2VSDKError(RuntimeError):
    pass


def run_i2v_async_then_wait(
    api_key: str,
    model: str,
    prompt: str,
    img_url: str,
    resolution: str,
    duration: int,
    audio_url: str | None = None,
    audio: bool | None = None,
    prompt_extend: bool = True,
    watermark: bool | None = None,
    seed: int = 12345,
) -> str:
    """
    先 async_call 获取 task，再 wait 等待完成，返回 video_url。
    """
    rsp = VideoSynthesis.async_call(
        api_key=api_key,
        model=model,
        prompt=prompt,
        img_url=img_url,
        audio_url=audio_url,
        resolution=resolution.upper(),
        duration=duration,
        audio=audio,
        prompt_extend=prompt_extend,
        watermark=watermark,
        negative_prompt="",
        seed=seed,
    )
    if rsp.status_code != HTTPStatus.OK:
        raise I2VSDKError(
            f"async_call failed status={rsp.status_code}, code={rsp.code}, message={rsp.message}"
        )

    waited = VideoSynthesis.wait(task=rsp, api_key=api_key)
    if waited.status_code != HTTPStatus.OK:
        raise I2VSDKError(
            f"wait failed status={waited.status_code}, code={waited.code}, message={waited.message}"
        )
    url = getattr(waited.output, "video_url", None)
    if not url:
        raise I2VSDKError("任务完成但未返回 video_url")
    return url

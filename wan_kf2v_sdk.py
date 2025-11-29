"""
通义万相首尾帧生视频（kf2v）：使用 DashScope SDK 的 async_call + wait。
"""
import dashscope
from dashscope import VideoSynthesis
from http import HTTPStatus

# 北京地域；如需新加坡请改为 https://dashscope-intl.aliyuncs.com/api/v1
dashscope.base_http_api_url = "https://dashscope.aliyuncs.com/api/v1"


class KF2VSDKError(RuntimeError):
    pass


def run_kf2v_async_then_wait(
    api_key: str,
    model: str,
    prompt: str,
    first_frame_url: str,
    last_frame_url: str,
    resolution: str = "720P",
    prompt_extend: bool = True,
    watermark: bool | None = None,
    seed: int = 12345,
) -> str:
    rsp = VideoSynthesis.async_call(
        api_key=api_key,
        model=model,
        prompt=prompt,
        first_frame_url=first_frame_url,
        last_frame_url=last_frame_url,
        resolution=resolution.upper(),
        prompt_extend=prompt_extend,
        watermark=watermark,
        negative_prompt="",
        seed=seed,
    )
    if rsp.status_code != HTTPStatus.OK:
        raise KF2VSDKError(
            f"async_call failed status={rsp.status_code}, code={rsp.code}, message={rsp.message}"
        )

    waited = VideoSynthesis.wait(task=rsp, api_key=api_key)
    if waited.status_code != HTTPStatus.OK:
        raise KF2VSDKError(
            f"wait failed status={waited.status_code}, code={waited.code}, message={waited.message}"
        )
    url = getattr(waited.output, "video_url", None)
    if not url:
        raise KF2VSDKError("任务完成但未返回 video_url")
    return url

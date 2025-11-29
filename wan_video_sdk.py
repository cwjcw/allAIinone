import os
from http import HTTPStatus

import dashscope
from dashscope import VideoSynthesis
from dotenv import load_dotenv

# 默认参数，可按需修改
DEFAULT_MODEL = "wan2.5-t2v-preview"
DEFAULT_PROMPT = (
    "一幅史诗级可爱的场景。一只小巧可爱的卡通小猫将军，身穿细节精致的金色盔甲，"
    "头戴一个稍大的头盔，勇敢地站在悬崖上。他骑着一匹虽小但英勇的战马，说："
    "“青海长云暗雪山，孤城遥望玉门关。黄沙百战穿金甲，不破楼兰终不还。”。"
    "悬崖下方，一支由老鼠组成的、数量庞大、无穷无尽的军队正带着临时制作的武器向前冲锋。"
    "这是一个戏剧性的、大规模的战斗场景，灵感来自中国古代的战争史诗。远处的雪山上空，天空乌云密布。"
    "整体氛围是“可爱”与“霸气”的搞笑和史诗般的融合。"
)
DEFAULT_AUDIO_URL = "https://help-static-aliyun-doc.aliyuncs.com/file-manage-files/zh-CN/20250923/hbiayh/%E4%BB%8E%E5%86%9B%E8%A1%8C.mp3"
DEFAULT_SIZE = "832*480"
DEFAULT_DURATION = 10
DEFAULT_SEED = 12345

# 若使用新加坡地域，取消注释下行
# dashscope.base_http_api_url = "https://dashscope-intl.aliyuncs.com/api/v1"


def load_api_key() -> str:
    load_dotenv()
    api_key = os.getenv("DASHSCOPE_API_KEY")
    if not api_key:
        raise RuntimeError("请在 .env 中设置 DASHSCOPE_API_KEY")
    return api_key


def sample_sync_call_t2v(
    model: str = DEFAULT_MODEL,
    prompt: str = DEFAULT_PROMPT,
    audio_url: str = DEFAULT_AUDIO_URL,
    size: str = DEFAULT_SIZE,
    duration: int = DEFAULT_DURATION,
    seed: int = DEFAULT_SEED,
    watermark: bool = False,
    prompt_extend: bool = True,
) -> None:
    api_key = load_api_key()
    print("please wait...")
    rsp = VideoSynthesis.call(
        api_key=api_key,
        model=model,
        prompt=prompt,
        audio_url=audio_url,
        size=size,
        duration=duration,
        negative_prompt="",
        prompt_extend=prompt_extend,
        watermark=watermark,
        seed=seed,
        # 如需音频输出，打开下行
        # audio=True,
    )
    print(rsp)
    if rsp.status_code == HTTPStatus.OK:
        print("video_url:", rsp.output.video_url)
    else:
        print(f"Failed, status_code: {rsp.status_code}, code: {rsp.code}, message: {rsp.message}")
        print("请参考文档：https://help.aliyun.com/zh/model-studio/developer-reference/error-code")


if __name__ == "__main__":
    sample_sync_call_t2v()

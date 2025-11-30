# ALL AI IN ONE

一个整合对话、画图、视频和用户管理的本地 Web 控制台，基于 OpenRouter 和通义万相（dashscope）。内置注册/登录、角色权限、余额扣费、近 3 天对话历史、头像与收款码展示等。

## 主要功能
- 对话：支持多轮对话、Markdown 渲染、Ctrl+Enter 发送、近 3 天历史侧栏一键查看。
- 画图：多模型图像生成，支持最多 4 张本地/URL 参考图，下载预览。
- 视频：文生视频、图生视频、首尾帧视频，基于通义万相 SDK。
- 用户体系：注册/登录，角色（free/paid/vip/admin）；付费用户图片/编辑扣 1 元，视频扣 3 元，VIP/管理员不限，免费仅对话。
- 用户中心：头像上传（存储于 `media/avatars`）、余额与权限卡片展示，充值/联系二维码（`media/paycode.jpg`、`media/wechat.jpg`），管理员可手工调整角色和余额。
- 上传接口：图片、音频上传后返回可访问 URL。
- 模型缓存：OpenRouter 模型列表可刷新并本地缓存。

## 快速开始
1. 准备环境变量文件（见下方 `.env.example`）。
2. 安装依赖：`pip install -r requirements.txt`。
3. 启动服务：`uvicorn server:app --reload`。
4. 访问 `http://localhost:8000/web/index.html`，首次注册的账号自动成为管理员。

## 环境变量示例
在项目根目录创建 `.env`（或使用 `.env.example` 拷贝）：
```
OPENROUTER_API_KEY=your_openrouter_key
DASHSCOPE_API_KEY=your_dashscope_key
SECRET_KEY=change-me    # 用于登录 Token 签名
OPENROUTER_SITE_URL=http://localhost   # 可选，传给 OpenRouter 作为 Referer
OPENROUTER_APP_TITLE=ALL AI IN ONE     # 可选，传给 OpenRouter 作为应用名
```

## 关键路径说明
- 后端入口：`server.py`（FastAPI，含鉴权、计费、历史、上传等）
- 前端页面：`public/`（index、chat、image、video、user 等页面）
- 用户数据：`data/users.json`（自动生成）
- 对话历史：`data/chat_history.json`（自动生成，保留 3 天）
- 媒体目录：`media/`（包含 `avatars/` 头像目录，已加入 `.gitignore`）

## 常见问题
- 登录后 401：清除浏览器缓存重新登录，确保 `SECRET_KEY` 未频繁变更。
- OpenRouter/Dashscope 报错：检查 API Key 是否正确，或查看返回的完整 `detail` 提示。
- 图片生成无结果：若模型返回无 `images`，错误会包含模型的 `reasoning` 片段，按提示调整输入或更换模型。

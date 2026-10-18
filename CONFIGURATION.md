# 运行配置

项目默认以离线模式启动，不需要任何外部 API。

## 基础配置

- `FASHIONAI_MODE`：`offline`（默认）或 `online`
- `FASHIONAI_HOST`：监听地址，默认 `127.0.0.1`
- `FASHIONAI_PORT`：监听端口，默认 `5000`
- `FASHIONAI_CORS_ORIGINS`：允许访问后端的前端来源，多个值用逗号分隔

## 可选模型配置

在线模式需要在运行环境中注入模型凭据，不要将真实凭据写入仓库：

- `ZHIPU_API_KEY`：文本模型凭据
- `DOUBAO_API_KEY`：视觉模型凭据

在线模式启动前安装 `requirements-ai.txt`，并将可选的本地嵌入模型放在 `bge-small-zh` 目录。

# FashionAI 服装电商智能客服

面向服装零售场景的智能客服与推荐系统。项目提供 Vue 3 交互界面、Flask API、流式对话、知识库检索、尺码建议、商品推荐、订单与售后工作流，并支持可选的大模型和视觉模型能力。

项目默认使用离线模式，无需任何 API 凭据即可运行核心演示；配置有效模型服务后，可切换到多智能体增强模式。

仓库中的商品目录、价格、库存和图片仅用于功能演示，不对应真实店铺或交易信息。

## 核心能力

- 离线可用：本地规则引擎与知识库检索保证 API 失效时仍能完成核心演示
- 多智能体编排：在线模式包含 Planner、Executor、Critic 与 ReAct 推理流程
- RAG 检索：支持业务知识、尺码说明和退换货政策查询
- 业务工作流：商品咨询、尺码推荐、订单查询和售后状态管理
- 流式交互：后端通过 SSE 输出回复，前端实时渲染
- 多模态扩展：配置视觉模型后可启用服装图片分析与穿搭建议
- 安全配置：凭据只从运行环境读取，限制 CORS 来源和上传大小

## 技术栈

Python、Flask、Vue 3、TypeScript、Vite、Element Plus、LangChain、BGE、TF-IDF、SSE

## 架构

~~~text
Vue 3 Web
    │ REST / SSE
    ▼
Flask API
    ├─ OfflineAssistant：规则、尺码计算、知识检索
    └─ Online Agents：Planner → Executor → Critic / ReAct
            ├─ RAG 知识库
            ├─ 商品与业务工具
            └─ 可选文本与视觉模型
~~~

## 项目结构

~~~text
.
├─ run.py                 # 统一启动入口，默认离线模式
├─ offline_api.py         # 无外部服务依赖的 Flask API
├─ offline_engine.py      # 可测试的本地客服引擎
├─ api.py                 # 完整在线 API 与业务工作流
├─ app.py                 # 智能体、RAG、工具和视觉能力
├─ knowledge/             # 商品、尺码、穿搭和售后知识
├─ fashionai-vue/         # Vue 3 前端
├─ tests/                 # 离线引擎与 API 测试
├─ requirements.txt       # 离线运行依赖
└─ requirements-ai.txt    # 在线增强能力依赖
~~~

## 快速开始

推荐 Python 3.10 或 3.11、Node.js 20 以上版本。

### 后端

~~~powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python run.py
~~~

默认启动 http://127.0.0.1:5000。健康检查：GET /health。

### 前端

~~~powershell
cd fashionai-vue
npm install
npm run dev
~~~

如需修改后端地址，可设置 VITE_API_BASE_URL。

## 在线增强模式

先安装扩展依赖：

~~~powershell
python -m pip install -r requirements-ai.txt
~~~

在运行环境中配置模型凭据并设置 FASHIONAI_MODE=online。详细配置见 [CONFIGURATION.md](CONFIGURATION.md)。真实凭据、订单数据、日志、数据库、对话历史、本地模型权重和构建产物均不会进入版本库。

## 测试

~~~powershell
python -m pytest -q
cd fashionai-vue
npm run build
~~~

## API

| 接口 | 方法 | 离线模式 | 说明 |
|---|---|---:|---|
| /health | GET | 是 | 服务状态与可用能力 |
| /chat | POST | 是 | 普通对话 |
| /chat/stream | POST | 是 | SSE 流式对话 |
| /chat/clear | POST | 是 | 清理会话状态 |
| /api/products | GET | 是 | 商品目录 |
| /chat/image | POST | 否 | 图片理解 |
| /chat/outfit_recommend | POST | 否 | 视觉穿搭推荐 |
| /query_order | POST | 在线版 | 订单查询 |
| /after_sales/* | 多种 | 在线版 | 售后工作流 |

## 已知限制

- 离线引擎以稳定演示和可测试性为目标，不替代生成式模型的开放问答能力。
- 图片理解和搭配图生成依赖外部视觉模型，凭据失效时会返回明确的能力不可用状态。
- 完整在线模块仍保留在 app.py 与 api.py，后续可继续拆分为独立领域服务。

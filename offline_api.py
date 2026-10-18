"""FashionAI 离线演示 API：不依赖任何外部模型服务。"""
from __future__ import annotations

import json
import os
import re
from pathlib import Path

from flask import Flask, Response, jsonify, request, stream_with_context
from flask_cors import CORS

from offline_engine import OfflineAssistant

ROOT = Path(__file__).resolve().parent
assistant = OfflineAssistant(ROOT)
sessions: dict[str, dict] = {}


def _allowed_origins() -> list[str]:
    raw = os.getenv("FASHIONAI_CORS_ORIGINS", "http://127.0.0.1:5173,http://localhost:5173")
    return [item.strip() for item in raw.split(",") if item.strip()]


def _load_products() -> list[dict]:
    path = ROOT / "knowledge" / "product_info.txt"
    if not path.exists():
        return []
    text = path.read_text(encoding="utf-8", errors="ignore")
    products = []
    for block in re.split(r"\n(?=\d+\.\s*商品名称：)", text):
        name = re.search(r"商品名称：([^\n]+)", block)
        if not name:
            continue
        def field(label: str) -> str:
            match = re.search(rf"{re.escape(label)}：([^\n]+)", block)
            return match.group(1).strip() if match else ""
        products.append({
            "name": name.group(1).strip(),
            "price": field("价格"),
            "color": field("颜色"),
            "size": field("尺码"),
            "material": field("材质"),
            "features": field("卖点"),
            "image_url": field("图片"),
        })
    return products


def create_app() -> Flask:
    app = Flask(__name__)
    app.config["MAX_CONTENT_LENGTH"] = 5 * 1024 * 1024
    CORS(app, resources={r"/*": {"origins": _allowed_origins()}})

    @app.get("/health")
    def health():
        return jsonify({
            "status": "ok",
            "mode": "offline",
            "capabilities": ["chat", "size_advice", "knowledge_search", "products"],
        })

    @app.get("/api/products")
    def products():
        return jsonify({"success": True, "products": _load_products()})

    @app.post("/chat")
    def chat():
        payload = request.get_json(silent=True) or {}
        message = str(payload.get("message", "")).strip()
        if not message:
            return jsonify({"success": False, "message": "message 不能为空"}), 400
        session_id = str(payload.get("session_id", "default"))[:80]
        profile = sessions.setdefault(session_id, {})
        supplied_profile = payload.get("user_info")
        if isinstance(supplied_profile, dict):
            profile.update(supplied_profile)
        return jsonify({"success": True, "response": assistant.reply(message, profile), "mode": "offline"})

    @app.post("/chat/stream")
    def chat_stream():
        payload = request.get_json(silent=True) or {}
        message = str(payload.get("message", "")).strip()
        if not message:
            return jsonify({"success": False, "message": "message 不能为空"}), 400
        session_id = str(payload.get("session_id", "default"))[:80]
        profile = sessions.setdefault(session_id, {})
        supplied_profile = payload.get("user_info")
        if isinstance(supplied_profile, dict):
            profile.update(supplied_profile)
        reply = assistant.reply(message, profile)

        def generate():
            for index in range(0, len(reply), 12):
                yield f"data: {json.dumps({'chunk': reply[index:index + 12], 'done': False}, ensure_ascii=False)}\n\n"
            yield f"data: {json.dumps({'done': True, 'mode': 'offline'}, ensure_ascii=False)}\n\n"

        return Response(stream_with_context(generate()), mimetype="text/event-stream")

    @app.post("/chat/clear")
    def clear_chat():
        payload = request.get_json(silent=True) or {}
        sessions.pop(str(payload.get("session_id", "default"))[:80], None)
        return jsonify({"success": True})

    @app.post("/chat/image")
    @app.post("/chat/outfit_recommend")
    def image_unavailable():
        return jsonify({
            "success": False,
            "mode": "offline",
            "message": "图片理解需要配置可用的视觉模型服务；文字客服功能仍可正常使用。",
        }), 503

    return app


app = create_app()


if __name__ == "__main__":
    app.run(
        host=os.getenv("FASHIONAI_HOST", "127.0.0.1"),
        port=int(os.getenv("FASHIONAI_PORT", "5000")),
        debug=False,
    )

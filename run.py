"""统一启动入口；默认离线，显式设置 FASHIONAI_MODE=online 才加载完整模型栈。"""
from __future__ import annotations

import os


def load_app():
    mode = os.getenv("FASHIONAI_MODE", "offline").strip().lower()
    if mode == "online":
        from api import app
        return app
    from offline_api import app
    return app


app = load_app()


if __name__ == "__main__":
    app.run(
        host=os.getenv("FASHIONAI_HOST", "127.0.0.1"),
        port=int(os.getenv("FASHIONAI_PORT", "5000")),
        debug=False,
        use_reloader=False,
    )

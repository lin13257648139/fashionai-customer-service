"""无需外部 API 的本地客服降级引擎。"""
from __future__ import annotations

import re
from pathlib import Path


class OfflineAssistant:
    """提供尺码、退换货、商品与穿搭建议，保证无 API 时仍可演示。"""

    def __init__(self, root: Path | None = None) -> None:
        self.root = root or Path(__file__).resolve().parent
        self.knowledge_dir = self.root / "knowledge"

    @staticmethod
    def recommend_size(height: int | None, weight: int | None) -> str | None:
        if not height or not weight:
            return None
        if height < 160:
            return "XS" if weight < 50 else "S" if weight < 60 else "M"
        if height < 170:
            return "S" if weight < 55 else "M" if weight < 65 else "L"
        if height < 180:
            return "M" if weight < 65 else "L" if weight < 75 else "XL"
        return "L" if weight < 75 else "XL" if weight < 90 else "XXL"

    @staticmethod
    def _extract_number(message: str, labels: tuple[str, ...]) -> int | None:
        label_group = "|".join(map(re.escape, labels))
        patterns = (
            rf"(?:{label_group})\s*(?:是|为|[:：])?\s*(\d{{2,3}})",
            rf"(\d{{2,3}})\s*(?:cm|厘米|斤|kg|公斤)",
        )
        for pattern in patterns:
            match = re.search(pattern, message, re.IGNORECASE)
            if match:
                return int(match.group(1))
        return None

    def _size_reply(self, message: str, profile: dict) -> str:
        height = profile.get("height") or self._extract_number(message, ("身高",))
        weight = profile.get("weight") or self._extract_number(message, ("体重",))
        if weight and "斤" in message and "kg" not in message.lower() and "公斤" not in message:
            weight = round(weight / 2)
        size = self.recommend_size(int(height) if height else None, int(weight) if weight else None)
        if not size:
            return "请补充身高（cm）和体重（kg），我可以先给出基础尺码建议；具体还要结合版型和商品尺码表确认。"
        return f"按身高 {height}cm、体重 {weight}kg 估算，建议先试 {size} 码。修身款可加一码，宽松款可按常规尺码选择。"

    def _search_knowledge(self, message: str) -> str | None:
        query_chars = {char for char in message if "\u4e00" <= char <= "\u9fff"}
        best_score, best_text = 0, ""
        for path in self.knowledge_dir.glob("*.txt"):
            text = path.read_text(encoding="utf-8", errors="ignore")
            for paragraph in re.split(r"\n\s*\n", text):
                cleaned = " ".join(line.strip() for line in paragraph.splitlines() if line.strip())
                if len(cleaned) < 8:
                    continue
                score = len(query_chars & {c for c in cleaned if "\u4e00" <= c <= "\u9fff"})
                if score > best_score:
                    best_score, best_text = score, cleaned
        return best_text[:360] if best_score >= 2 else None

    def reply(self, message: str, profile: dict | None = None) -> str:
        text = str(message or "").strip()
        profile = profile or {}
        if not text:
            return "请输入您想咨询的问题。"
        if any(word in text for word in ("尺码", "身高", "体重", "多大码")):
            return self._size_reply(text, profile)
        if any(word in text for word in ("退货", "换货", "退款", "售后")):
            result = self._search_knowledge(text)
            return result or "请保持商品完好并准备订单信息，再通过售后入口提交申请；实际时效以店铺规则为准。"
        if any(word in text for word in ("推荐", "搭配", "穿什么", "款式", "面料")):
            result = self._search_knowledge(text)
            return result or "可以告诉我使用场景、偏好颜色、预算和尺码，我会据此给出搭配建议。"
        if any(word in text.lower() for word in ("你好", "您好", "hello", "hi")):
            return "您好，我是 FashionAI。离线模式下可以处理尺码建议、穿搭咨询、商品知识和退换货规则。"
        result = self._search_knowledge(text)
        if result:
            return result
        return "当前处于离线演示模式。您可以咨询尺码、穿搭、商品材质或退换货规则；配置模型 API 后可启用多智能体推理。"

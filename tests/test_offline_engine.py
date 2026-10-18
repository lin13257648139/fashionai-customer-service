from pathlib import Path

from offline_engine import OfflineAssistant


def test_size_recommendation_is_available_without_api():
    assistant = OfflineAssistant(Path(__file__).resolve().parents[1])
    reply = assistant.reply("我身高175cm，体重70kg，穿多大码？")
    assert "L" in reply
    assert "175" in reply


def test_missing_size_profile_requests_details():
    assistant = OfflineAssistant(Path(__file__).resolve().parents[1])
    reply = assistant.reply("帮我推荐尺码")
    assert "身高" in reply
    assert "体重" in reply


def test_greeting_describes_offline_capabilities():
    assistant = OfflineAssistant(Path(__file__).resolve().parents[1])
    reply = assistant.reply("你好")
    assert "离线模式" in reply
    assert "尺码" in reply

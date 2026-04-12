"""Unit tests for MiniMax provider support in JoyAI-Image prompt rewriting."""

from __future__ import annotations

import os
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

# Ensure src is on the path
SRC_DIR = Path(__file__).resolve().parent.parent / "src"
sys.path.insert(0, str(SRC_DIR))

from infer_runtime.settings import MINIMAX_BASE_URL, _is_minimax_model


# ---------------------------------------------------------------------------
# _is_minimax_model detection
# ---------------------------------------------------------------------------

class TestIsMinimaxModel:
    def test_detects_minimax_m2_7(self):
        assert _is_minimax_model("MiniMax-M2.7") is True

    def test_detects_minimax_m2_7_highspeed(self):
        assert _is_minimax_model("MiniMax-M2.7-highspeed") is True

    def test_case_insensitive(self):
        assert _is_minimax_model("minimax-m2.7") is True
        assert _is_minimax_model("MINIMAX-M2.7") is True

    def test_rejects_openai_models(self):
        assert _is_minimax_model("gpt-5") is False
        assert _is_minimax_model("gpt-4o") is False

    def test_rejects_anthropic_models(self):
        assert _is_minimax_model("claude-3-5-sonnet") is False

    def test_rejects_other_models(self):
        assert _is_minimax_model("llama-3") is False


# ---------------------------------------------------------------------------
# MINIMAX_BASE_URL constant
# ---------------------------------------------------------------------------

class TestMinimaxBaseUrl:
    def test_base_url_uses_minimax_io(self):
        assert "api.minimax.io" in MINIMAX_BASE_URL

    def test_base_url_not_minimax_chat(self):
        assert "api.minimax.chat" not in MINIMAX_BASE_URL

    def test_base_url_is_v1_endpoint(self):
        assert MINIMAX_BASE_URL.endswith("/v1")


# ---------------------------------------------------------------------------
# Settings auto-configuration for MiniMax
# ---------------------------------------------------------------------------

class TestSettingsMinimaxAutoConfigure:
    """Test that load_settings auto-configures MiniMax when a MiniMax model is used."""

    def _make_settings(self, rewrite_model: str, env: dict) -> object:
        """Helper that patches os.environ and resolve_checkpoint_layout."""
        fake_layout = MagicMock()
        fake_layout.root = Path("/fake")
        fake_layout.transformer_ckpt = Path("/fake/transformer.pt")

        with patch.dict(os.environ, env, clear=False), \
             patch("infer_runtime.settings.resolve_checkpoint_layout", return_value=fake_layout), \
             patch("pathlib.Path.exists", return_value=True):
            from infer_runtime.settings import load_settings
            return load_settings(ckpt_root="/fake", rewrite_model=rewrite_model)

    def test_minimax_model_uses_minimax_api_key(self):
        env = {"MINIMAX_API_KEY": "sk-minimax-test", "OPENAI_API_KEY": "sk-openai-test"}
        settings = self._make_settings("MiniMax-M2.7", env)
        assert settings.openai_api_key == "sk-minimax-test"

    def test_minimax_model_sets_default_base_url(self):
        env = {"MINIMAX_API_KEY": "sk-minimax-test"}
        # Clear any override
        os.environ.pop("OPENAI_BASE_URL", None)
        settings = self._make_settings("MiniMax-M2.7", env)
        assert settings.openai_base_url == MINIMAX_BASE_URL

    def test_minimax_model_respects_custom_base_url(self):
        env = {"MINIMAX_API_KEY": "sk-minimax-test", "OPENAI_BASE_URL": "https://custom.minimax.io/v1"}
        settings = self._make_settings("MiniMax-M2.7", env)
        assert settings.openai_base_url == "https://custom.minimax.io/v1"

    def test_non_minimax_model_uses_openai_key(self):
        env = {"OPENAI_API_KEY": "sk-openai-test", "MINIMAX_API_KEY": "sk-minimax-test"}
        settings = self._make_settings("gpt-5", env)
        assert settings.openai_api_key == "sk-openai-test"

    def test_non_minimax_model_no_base_url_default(self):
        env = {"OPENAI_API_KEY": "sk-openai-test"}
        os.environ.pop("OPENAI_BASE_URL", None)
        settings = self._make_settings("gpt-5", env)
        assert settings.openai_base_url is None

    def test_default_model_is_gpt5(self):
        env = {"OPENAI_API_KEY": "sk-openai-test"}
        fake_layout = MagicMock()
        fake_layout.root = Path("/fake")
        fake_layout.transformer_ckpt = Path("/fake/transformer.pt")
        with patch.dict(os.environ, env, clear=False), \
             patch("infer_runtime.settings.resolve_checkpoint_layout", return_value=fake_layout), \
             patch("pathlib.Path.exists", return_value=True):
            from infer_runtime.settings import load_settings
            settings = load_settings(ckpt_root="/fake")
        assert settings.rewrite_model == "gpt-5"


# ---------------------------------------------------------------------------
# Temperature selection in prompt_rewrite
# ---------------------------------------------------------------------------

class TestPromptRewriteTemperature:
    """Verify temperature is set correctly for MiniMax and other models."""

    def _capture_temperature(self, model: str) -> float:
        """Call rewrite_prompt with a mock OpenAI client and capture the temperature used."""
        from infer_runtime.prompt_rewrite import rewrite_prompt

        captured = {}

        class FakeChoices:
            message = MagicMock()
            message.content = '{"Rewritten": "test prompt"}'

        class FakeResponse:
            choices = [FakeChoices()]

        def fake_create(**kwargs):
            captured["temperature"] = kwargs.get("temperature")
            return FakeResponse()

        fake_client = MagicMock()
        fake_client.chat.completions.create.side_effect = fake_create

        with patch("openai.OpenAI", return_value=fake_client):
            rewrite_prompt("test", None, model=model, api_key="sk-test", base_url=None)

        return captured.get("temperature", -999)

    def test_minimax_m2_7_temperature_is_one(self):
        temp = self._capture_temperature("MiniMax-M2.7")
        assert temp == 1.0, f"Expected 1.0, got {temp}"

    def test_minimax_highspeed_temperature_is_one(self):
        temp = self._capture_temperature("MiniMax-M2.7-highspeed")
        assert temp == 1.0, f"Expected 1.0, got {temp}"

    def test_minimax_temperature_is_not_zero(self):
        temp = self._capture_temperature("MiniMax-M2.7")
        assert temp != 0.0, "MiniMax does not accept temperature=0.0"

    def test_gpt5_temperature_is_one(self):
        temp = self._capture_temperature("gpt-5")
        assert temp == 1.0

    def test_other_model_temperature_is_zero(self):
        temp = self._capture_temperature("gpt-4o")
        assert temp == 0.0


# ---------------------------------------------------------------------------
# Integration test (skipped without MINIMAX_API_KEY)
# ---------------------------------------------------------------------------

MINIMAX_API_KEY = os.environ.get("MINIMAX_API_KEY")


@pytest.mark.skipif(not MINIMAX_API_KEY, reason="MINIMAX_API_KEY not set")
class TestMinimaxIntegration:
    """Live integration test against the MiniMax API."""

    def test_minimax_chat_completion(self):
        """Verify MiniMax OpenAI-compatible endpoint returns a valid response."""
        from openai import OpenAI

        client = OpenAI(api_key=MINIMAX_API_KEY, base_url=MINIMAX_BASE_URL)
        response = client.chat.completions.create(
            model="MiniMax-M2.7",
            messages=[{"role": "user", "content": 'Reply with JSON: {"Rewritten": "test passed"}'}],
            temperature=1.0,
            max_tokens=50,
        )
        content = response.choices[0].message.content or ""
        assert "Rewritten" in content or len(content) > 0

    def test_rewrite_prompt_with_minimax(self):
        """End-to-end rewrite_prompt call using MiniMax."""
        from infer_runtime.prompt_rewrite import rewrite_prompt

        result = rewrite_prompt(
            "make the sky blue",
            None,
            model="MiniMax-M2.7",
            api_key=MINIMAX_API_KEY,
            base_url=MINIMAX_BASE_URL,
        )
        assert isinstance(result, str) and len(result) > 0

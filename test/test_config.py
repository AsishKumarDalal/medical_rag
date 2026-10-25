"""H1: configuration comes from the environment and fails loudly when wrong."""

from __future__ import annotations

import pytest

from harness.config import Config, ConfigError, load_env_file

BASE_ENV = {
    "LLM_BASE_URL": "https://example.test/v1/",
    "LLM_API_KEY": "secret-key",
    "LLM_MODEL": "some-model",
}


def test_defaults_apply_and_base_url_is_normalized():
    config = Config.from_env(BASE_ENV)

    assert config.base_url == "https://example.test/v1"
    assert config.max_tokens == 4096
    assert config.context_window == 50_000
    assert config.max_iterations == 12
    assert config.temperature is None


def test_missing_required_variable_is_named():
    with pytest.raises(ConfigError, match="LLM_MODEL"):
        Config.from_env({"LLM_BASE_URL": "https://x.test/v1", "LLM_API_KEY": "k"})


def test_completion_budget_must_fit_inside_the_context_window():
    with pytest.raises(ConfigError, match="leave room for the prompt"):
        Config.from_env({**BASE_ENV, "LLM_MAX_TOKENS": "50000"})


def test_non_integer_setting_is_reported():
    with pytest.raises(ConfigError, match="LLM_MAX_TOKENS"):
        Config.from_env({**BASE_ENV, "LLM_MAX_TOKENS": "lots"})


def test_api_key_is_not_in_repr():
    assert "secret-key" not in repr(Config.from_env(BASE_ENV))


def test_env_file_supports_comments_quotes_and_export(tmp_path, monkeypatch):
    for key in ("LLM_MODEL", "HARNESS_MAX_ITERATIONS"):
        monkeypatch.delenv(key, raising=False)
    env_file = tmp_path / ".env"
    env_file.write_text(
        "# comment\n"
        "\n"
        'LLM_MODEL="quoted-model"\n'
        "export HARNESS_MAX_ITERATIONS=4\n",
        encoding="utf-8",
    )

    load_env_file(env_file)

    import os

    assert os.environ["LLM_MODEL"] == "quoted-model"
    assert os.environ["HARNESS_MAX_ITERATIONS"] == "4"


def test_env_file_does_not_override_existing_values(tmp_path, monkeypatch):
    monkeypatch.setenv("LLM_MODEL", "from-shell")
    env_file = tmp_path / ".env"
    env_file.write_text("LLM_MODEL=from-file\n", encoding="utf-8")

    load_env_file(env_file)

    import os

    assert os.environ["LLM_MODEL"] == "from-shell"


def test_malformed_env_line_is_an_error(tmp_path):
    env_file = tmp_path / ".env"
    env_file.write_text("this is not an assignment\n", encoding="utf-8")

    with pytest.raises(ConfigError, match="expected KEY=VALUE"):
        load_env_file(env_file)

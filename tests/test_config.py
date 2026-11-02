import huggingface_hub
import pytest

import config


def test_adapter_path_local_then_hub(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "OUTPUTS", tmp_path / "outputs")
    hub = tmp_path / "hub"
    (hub / "lang").mkdir(parents=True)
    calls = []
    monkeypatch.setattr(huggingface_hub, "snapshot_download", lambda repo, allow_patterns: calls.append((repo, allow_patterns)) or str(hub))
    assert config.adapter_path("lang", "qwen2.5-7b") == hub / "lang"
    assert calls == [(config.HUB_ADAPTERS["qwen2.5-7b"], "lang/*")]
    with pytest.raises(FileNotFoundError):
        config.adapter_path("format", "qwen2.5-7b")
    local = config.lora_dir("lang", "qwen2.5-7b")
    local.mkdir(parents=True)
    assert config.adapter_path("lang", "qwen2.5-7b") == local  # a local fine-tune wins
    assert len(calls) == 2

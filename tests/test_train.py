from pathlib import Path

import pytest

from tempus.schema import Example, Extraction
from tempus.target import FORMATS
from tempus.train import load_config, lora_targets, split, token_row

CONFIGS = Path(__file__).parent.parent / "configs"


@pytest.mark.parametrize("path", sorted(CONFIGS.glob("*.yaml")), ids=lambda p: p.name)
def test_every_config_loads_with_typed_fields(path):
    config = load_config(path)
    assert config.name == path.stem
    assert config.loader in ("FastModel", "FastLanguageModel") and config.format in FORMATS
    assert isinstance(config.learning_rate, float) and isinstance(config.data, Path)


def test_config_with_a_missing_or_unknown_key_is_refused(tmp_path):
    path = tmp_path / "run.yaml"
    path.write_text(
        "model: m\nloader: FastModel\ndata: d.jsonl\nformat: json\nlearning_rate: 1e-4\nepochs: 1\nsed: 0\n"
    )
    with pytest.raises(ValueError, match="expected keys"):
        load_config(path)


def test_target_modules_are_optional_and_read_as_a_tuple(tmp_path):
    path = tmp_path / "run.yaml"
    base = "model: m\nloader: FastModel\ndata: d.jsonl\nformat: compact\nlearning_rate: 2e-4\nepochs: 3\nseed: 0\n"
    path.write_text(base)
    assert load_config(path).target_modules == ()
    path.write_text(base + "target_modules: [q_proj, in_proj]\n")
    assert load_config(path).target_modules == ("q_proj", "in_proj")


def test_named_lora_targets_turn_every_unsloth_filter_on():
    named = lora_targets(("q_proj", "in_proj"))
    assert named["target_modules"] == ["q_proj", "in_proj"]
    assert all(value for key, value in named.items() if key.startswith("finetune_"))
    default = lora_targets(())
    assert default["target_modules"] is None and default["finetune_vision_layers"] is False


class CharTokenizer:
    """A chat template with a thinking block only in the generation prompt; ids are character codes."""

    def apply_chat_template(self, conversation, add_generation_prompt=False, **kwargs):
        return "<s>" + "".join(f"<{m['role']}>{m['content']}" for m in conversation) + (
            "<assistant><think></think>" if add_generation_prompt else "<end>\n"
        )

    def __call__(self, text, add_special_tokens):
        assert add_special_tokens is False
        return {"input_ids": [ord(c) for c in text]}


def test_token_row_is_the_inference_prompt_then_the_reply_with_loss_only_on_the_reply():
    example = Example("x", "implicit", "dinner at 7?", (Extraction("7", 1140),))
    row = token_row(CharTokenizer(), example, "compact", "<end>")
    prompt = "<s><user>dinner at 7?<assistant><think></think>"
    assert row["input_ids"] == [ord(c) for c in prompt + "7=19:00<end>"]
    assert row["completion_mask"] == [0] * len(prompt) + [1] * len("7=19:00<end>")


def test_split_is_deterministic_disjoint_and_complete():
    examples = [Example(f"e{i}", "none", f"message {i}", ()) for i in range(59)]
    train, val = split(examples, seed=0)
    assert (train, val) == split(examples, seed=0)
    assert len(val) == 6 and {e.id for e in train} | {e.id for e in val} == {e.id for e in examples}
    assert split(examples, seed=1)[1] != val


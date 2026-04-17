"""Model zoo invariants: every model has a sensible max_seq_len."""
from hetero_cost_model.model_zoo import MODELS, MODEL_BY_NAME


def test_every_model_has_max_seq_len():
    for spec in MODELS:
        assert spec.max_seq_len > 0, f"{spec.name} has non-positive max_seq_len"


def test_bert_max_seq_len_is_512():
    """BERT max_position_embeddings is 512; going over triggers a
    position-embedding index error, not a real OOM."""
    assert MODEL_BY_NAME["bert-base"].max_seq_len == 512
    assert MODEL_BY_NAME["bert-large"].max_seq_len == 512


def test_gpt2_max_seq_len_is_1024():
    """GPT-2's learned positional embedding table is of size 1024."""
    for name in ("gpt2-small", "gpt2-medium", "gpt2-large"):
        assert MODEL_BY_NAME[name].max_seq_len == 1024


def test_t5_max_seq_len_at_least_covers_grid():
    """T5 uses relative position bias so long seqs work, but our current
    grid maxes at 1024 so just assert the spec honors that."""
    assert MODEL_BY_NAME["t5-small"].max_seq_len >= 1024

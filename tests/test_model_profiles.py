from turboinfer.model_profiles import get_model_profile


def test_qwen2_5_profile_matches_project_shape() -> None:
    profile = get_model_profile("qwen2.5-0.5b")

    assert profile.hidden_size == 896
    assert profile.num_q_heads == 14
    assert profile.num_kv_heads == 2
    assert profile.head_dim == 64
    assert profile.gqa_group_size == 7
    assert profile.uses_gqa


def test_qwen3_profile_matches_project_shape() -> None:
    profile = get_model_profile("qwen3-0.6b")

    assert profile.hidden_size == 1024
    assert profile.num_q_heads == 16
    assert profile.num_kv_heads == 8
    assert profile.head_dim == 128
    assert profile.gqa_group_size == 2
    assert profile.uses_gqa

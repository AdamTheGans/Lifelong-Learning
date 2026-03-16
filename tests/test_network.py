import torch

from lifelong_learning.agents.ppo.network import CNNActorCritic, CONTEXT_CODE_DIM


def test_zero_context_code_keeps_neuromodulation_neutral():
    model = CNNActorCritic((21, 8, 8), 3)
    zero_code = torch.zeros(CONTEXT_CODE_DIM)

    model.set_context_code(zero_code)

    assert torch.allclose(model.neuro_mask, torch.ones_like(model.neuro_mask))


def test_zero_context_matches_unmasked_forward_pass():
    torch.manual_seed(0)
    model = CNNActorCritic((21, 8, 8), 3)
    obs = torch.randn(2, 21, 8, 8)

    model.set_context_code(torch.zeros(CONTEXT_CODE_DIM))
    zero_logits, zero_value = model.forward(obs)
    clear_logits, clear_value = model.forward_with_mask(obs, torch.ones_like(model.neuro_mask))

    assert torch.allclose(zero_logits, clear_logits)
    assert torch.allclose(zero_value, clear_value)

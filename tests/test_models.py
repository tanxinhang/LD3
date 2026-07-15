import torch

from ld3.models import PhysicsGuidedCrossAttention, TFOnlyEstimator


def test_model_shapes_and_null_token() -> None:
    batch, n, m, paths = 2, 16, 8, 4
    tf_input = torch.randn(batch, 3, n, m)
    tokens = torch.zeros(batch, paths, 7)
    tokens[:, :, 0] = torch.rand(batch, paths) * 6
    tokens[:, :, 1] = torch.rand(batch, paths) * 4 - 2
    tokens[:, :, 2] = 0.25
    tokens[:, :, 3] = 1.0
    tokens[:, :, 6] = 1.0
    valid = torch.ones(batch, paths, dtype=torch.bool)

    model = PhysicsGuidedCrossAttention(hidden_dim=16, token_dim=16)
    output, diagnostics = model(tf_input, tokens, valid)
    assert output.shape == (batch, 2, n, m)
    assert diagnostics["attention"].shape == (batch, n * m, paths + 1)
    assert torch.all(torch.isfinite(output))
    assert torch.allclose(
        diagnostics["attention"].sum(dim=-1),
        torch.ones(batch, n * m),
        atol=1e-5,
    )


def test_tf_only_shape() -> None:
    x = torch.randn(2, 3, 16, 8)
    output = TFOnlyEstimator(hidden_dim=16)(x)
    assert output.shape == (2, 2, 16, 8)

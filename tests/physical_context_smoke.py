import torch

from wan_video_action.models.physical_context import (
    PhysicalContextEncoder,
    PhysicalResidualAdapterBank,
)
from wan_video_action.pipelines.wan_video_action import WanVideoUnit_PhysicalContextEmbedder


def test_physical_context_encoder():
    encoder = PhysicalContextEncoder(
        context_dim=32,
        model_dim=64,
        num_tokens=2,
        hidden_dim=48,
    )
    token_emb, mod_emb = encoder(
        batch_size=3,
        target_groups=5,
        dtype=torch.float32,
        device="cpu",
    )
    assert token_emb.shape == (3, 2, 64)
    assert mod_emb.shape == (3, 5, 64)

    loss = token_emb.sum() + mod_emb.sum()
    loss.backward()
    assert encoder.default_context.grad is not None


def test_physical_context_unit():
    class DummyPipe:
        device = torch.device("cpu")
        torch_dtype = torch.float32
        physical_context_mode = "both"

        def __init__(self):
            self.physical_context_encoder = PhysicalContextEncoder(
                context_dim=16,
                model_dim=32,
                num_tokens=1,
                hidden_dim=32,
            )

        def load_models_to_device(self, _model_names):
            return None

    unit = WanVideoUnit_PhysicalContextEmbedder()
    pipe = DummyPipe()
    action = torch.zeros(4, 9, 14)
    context = torch.randn(4, 1, 16)
    outputs = unit.process(pipe, physical_context=context, action=action, num_frames=9)
    assert outputs["physical_context_emb"].shape == (4, 1, 32)
    assert outputs["physical_mod_emb"].shape == (4, 3, 32)


def test_physical_residual_adapter_bank():
    bank = PhysicalResidualAdapterBank(
        dim=64,
        num_layers=4,
        rank=8,
        layers=[1, 3],
    )
    x = torch.randn(2, 10, 64, requires_grad=True)
    assert torch.equal(bank(0, x), x)

    y = bank(1, x)
    assert torch.allclose(y, x)
    y.sum().backward()
    assert bank.adapters["1"].gate.grad is not None


if __name__ == "__main__":
    test_physical_context_encoder()
    test_physical_context_unit()
    test_physical_residual_adapter_bank()
    print("BWM_PHYSICAL_CONTEXT_SMOKE_OK")

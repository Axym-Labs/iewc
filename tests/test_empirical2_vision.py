import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from iewc.empirical2_vision import evaluate


class FixedLogitModel(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        logits = torch.tensor([[0.5, 0.1, 2.0]], dtype=x.dtype, device=x.device)
        return logits.repeat(x.shape[0], 1)


def test_task_aware_evaluation_masks_out_unavailable_classes():
    dataset = TensorDataset(torch.zeros(4, 1), torch.zeros(4, dtype=torch.long))
    loader = DataLoader(dataset, batch_size=2)
    model = FixedLogitModel()

    class_incremental_acc = evaluate(model, loader, torch.device("cpu"))
    task_aware_acc = evaluate(model, loader, torch.device("cpu"), allowed_classes=[0, 1])

    assert class_incremental_acc == 0.0
    assert task_aware_acc == 1.0

import unittest

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from iewc.diagonal_regularization import (
    compute_diagonal_importance,
    diagonal_ewc_penalty,
)


def make_loader(batch_size=1):
    x = torch.tensor([[2.0, 0.0], [1.5, 0.5], [-2.0, 0.0], [-1.5, -0.5]])
    y = torch.tensor([0, 0, 1, 1])
    return DataLoader(TensorDataset(x, y), batch_size=batch_size, shuffle=False)


def make_model():
    model = nn.Linear(2, 2, bias=False)
    with torch.no_grad():
        model.weight.copy_(torch.tensor([[2.0, 0.5], [-2.0, -0.5]]))
    return model


def loss_output_fn(model, batch, reverse_output):
    x, y = batch
    output = model(x)
    loss_input = -output if reverse_output else output
    return nn.functional.cross_entropy(loss_input, y), output


class DiagonalRegularizationTests(unittest.TestCase):
    def test_iewc_matches_manual_diagonal_for_trainable_parameters(self):
        model = make_model()
        loader = make_loader()
        tau = 0.25
        result = compute_diagonal_importance(
            model=model,
            dataloader=loader,
            loss_output_fn=loss_output_fn,
            device=torch.device("cpu"),
            kind="iewc",
            tau=tau,
        )

        manual = torch.zeros_like(model.weight)
        scales = []
        for batch in loader:
            model.zero_grad(set_to_none=True)
            x, y = batch
            output = model(x)
            loss = nn.functional.cross_entropy(output, y)
            output_grad = torch.autograd.grad(loss, output, retain_graph=True)[0]
            scale = output_grad.detach().pow(2).sum()
            loss.backward()
            manual += model.weight.grad.detach().pow(2) / (scale + tau)
            scales.append(scale)
        manual /= len(loader.dataset)

        self.assertTrue(torch.allclose(result.importances["weight"], manual))
        self.assertTrue(torch.allclose(result.loss_scales, torch.stack(scales)))

    def test_weighting_modes_are_mean_normalized(self):
        for weighting in ("gss_residual", "fromp_trace"):
            result = compute_diagonal_importance(
                model=make_model(),
                dataloader=make_loader(),
                loss_output_fn=loss_output_fn,
                device=torch.device("cpu"),
                kind="iewc",
                tau=0.25,
                sample_weighting=weighting,
            )
            self.assertAlmostEqual(result.sample_weights.mean().item(), 1.0, places=5)
            self.assertEqual(result.sample_count, 4)

    def test_penalty_uses_saved_centers(self):
        model = make_model()
        result = compute_diagonal_importance(
            model=model,
            dataloader=make_loader(),
            loss_output_fn=loss_output_fn,
            device=torch.device("cpu"),
            kind="ef",
        )

        self.assertEqual(diagonal_ewc_penalty(model, result, torch.device("cpu")).item(), 0.0)
        with torch.no_grad():
            model.weight.add_(0.1)
        self.assertGreater(diagonal_ewc_penalty(model, result, torch.device("cpu")).item(), 0.0)


if __name__ == "__main__":
    unittest.main()

import types
import unittest

import torch
from torch import nn
from torch.utils.data import TensorDataset

from iewc.config import IEWCConfig
from iewc.low_rank import (
    LowRankIEWCPlugin,
    LowRankImportanceEstimator,
    _centered_covariance_eigenpairs,
)


def make_dataset():
    x = torch.tensor(
        [
            [2.0, 0.0],
            [1.5, 0.5],
            [-2.0, 0.0],
            [-1.5, -0.5],
        ]
    )
    y = torch.tensor([0, 0, 1, 1])
    task = torch.zeros(len(y), dtype=torch.long)
    return TensorDataset(x, y, task)


def make_model():
    model = nn.Linear(2, 2, bias=False)
    with torch.no_grad():
        model.weight.copy_(torch.tensor([[2.0, 0.5], [-2.0, -0.5]]))
    return model


def flattened_grads(model, dataset, criterion, optimizer, tau):
    rows = []
    scales = []
    for x, y, _ in dataset:
        optimizer.zero_grad()
        logits = model(x.unsqueeze(0))
        loss = criterion(logits, y.unsqueeze(0))
        logits_grad = torch.autograd.grad(loss, logits, retain_graph=True)[0]
        scale = logits_grad.detach().pow(2).sum()
        loss.backward()
        grad = torch.cat(
            [
                param.grad.detach().reshape(-1)
                for param in model.parameters()
                if param.grad is not None
            ]
        )
        rows.append(grad / torch.sqrt(scale + tau))
        scales.append(scale)
    return torch.stack(rows), torch.stack(scales)


class LowRankIEWCTests(unittest.TestCase):
    def test_low_rank_iewc_reconstructs_exact_small_covariance(self):
        model = make_model()
        dataset = make_dataset()
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
        tau = 0.25

        result = LowRankImportanceEstimator(
            kind="ief_low_rank", rank=4, tau=tau
        ).compute(
            model=model,
            criterion=criterion,
            optimizer=optimizer,
            dataset=dataset,
            device=torch.device("cpu"),
            batch_size=2,
        )

        rows, scales = flattened_grads(model, dataset, criterion, optimizer, tau)
        manual = rows.T @ rows / len(dataset)
        reconstructed = (
            result.eigenvectors.T
            @ torch.diag(result.eigenvalues)
            @ result.eigenvectors
        )

        self.assertEqual(result.sample_count, len(dataset))
        self.assertTrue(torch.allclose(result.loss_scales, scales))
        self.assertTrue(torch.allclose(reconstructed, manual, atol=1e-6))

    def test_low_rank_plus_diagonal_preserves_covariance_diagonal(self):
        model = make_model()
        dataset = make_dataset()
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
        tau = 0.25

        result = LowRankImportanceEstimator(
            kind="ief_low_rank_diag", rank=1, tau=tau
        ).compute(
            model=model,
            criterion=criterion,
            optimizer=optimizer,
            dataset=dataset,
            device=torch.device("cpu"),
            batch_size=2,
        )

        rows, _ = flattened_grads(model, dataset, criterion, optimizer, tau)
        manual = rows.T @ rows / len(dataset)
        low_rank = (
            result.eigenvectors.T
            @ torch.diag(result.eigenvalues)
            @ result.eigenvectors
        )
        hybrid = low_rank + torch.diag(result.residual_diagonal)

        self.assertTrue(torch.all(result.residual_diagonal >= 0))
        self.assertTrue(torch.allclose(torch.diag(hybrid), torch.diag(manual), atol=1e-6))
        self.assertTrue(torch.allclose(hybrid - torch.diag(torch.diag(hybrid)), low_rank - torch.diag(torch.diag(low_rank)), atol=1e-6))

    def test_diagonal_then_low_rank_uses_signed_residual_with_psd_surrogate(self):
        model = make_model()
        dataset = make_dataset()
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
        tau = 0.25

        result = LowRankImportanceEstimator(
            kind="ief_diag_low_rank", rank=2, tau=tau
        ).compute(
            model=model,
            criterion=criterion,
            optimizer=optimizer,
            dataset=dataset,
            device=torch.device("cpu"),
            batch_size=2,
        )

        rows, _ = flattened_grads(model, dataset, criterion, optimizer, tau)
        full_diagonal = rows.pow(2).mean(dim=0)
        low_rank_residual = (
            result.eigenvectors.T
            @ torch.diag(result.eigenvalues)
            @ result.eigenvectors
        )
        surrogate = torch.diag(result.residual_diagonal) + low_rank_residual

        self.assertTrue(torch.allclose(result.residual_diagonal, full_diagonal, atol=1e-6))
        self.assertTrue(torch.allclose(torch.diag(rows.T @ rows / len(dataset) - torch.diag(full_diagonal)), torch.zeros_like(full_diagonal), atol=1e-6))
        self.assertTrue(torch.all(torch.linalg.eigvalsh(surrogate) >= -1e-6))

    def test_correlation_low_rank_whitens_before_approximating_residual(self):
        model = make_model()
        dataset = make_dataset()
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
        tau = 0.25

        result = LowRankImportanceEstimator(
            kind="ief_corr_low_rank", rank=4, tau=tau
        ).compute(
            model=model,
            criterion=criterion,
            optimizer=optimizer,
            dataset=dataset,
            device=torch.device("cpu"),
            batch_size=2,
        )

        rows, _ = flattened_grads(model, dataset, criterion, optimizer, tau)
        manual = rows.T @ rows / len(dataset)
        full_diagonal = rows.pow(2).mean(dim=0)
        correction = (
            result.eigenvectors.T
            @ torch.diag(result.eigenvalues)
            @ result.eigenvectors
        )
        surrogate = torch.diag(result.residual_diagonal) + correction
        safe_diagonal = full_diagonal.clamp_min(1e-12)
        whitened_vectors = result.eigenvectors / torch.sqrt(safe_diagonal).unsqueeze(0)

        self.assertTrue(torch.allclose(result.residual_diagonal, full_diagonal, atol=1e-6))
        self.assertTrue(torch.allclose(surrogate, manual, atol=1e-5))
        self.assertTrue(
            torch.allclose(
                whitened_vectors @ whitened_vectors.T,
                torch.eye(result.rank),
                atol=1e-5,
            )
        )

    def test_centered_low_rank_spends_rank_on_variation_not_mean(self):
        rows = torch.tensor(
            [
                [10.0, 1.0],
                [10.0, -1.0],
                [10.0, 1.0],
                [10.0, -1.0],
            ]
        )

        eigvals, eigenvectors = _centered_covariance_eigenpairs(
            rows,
            rank=1,
            min_eigenvalue=1e-12,
        )

        self.assertTrue(torch.allclose(eigvals, torch.tensor([1.0]), atol=1e-6))
        self.assertLess(abs(float(eigenvectors[0, 0])), 1e-6)
        self.assertAlmostEqual(abs(float(eigenvectors[0, 1])), 1.0, places=6)

    def test_low_rank_plugin_adds_positive_penalty_after_parameter_drift(self):
        model = make_model()
        dataset = make_dataset()
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
        plugin = LowRankIEWCPlugin(
            ewc_lambda=10.0,
            rank=2,
            tau=0.25,
        )

        plugin.after_training_exp(
            types.SimpleNamespace(
                clock=types.SimpleNamespace(train_exp_counter=0),
                model=model,
                _criterion=criterion,
                optimizer=optimizer,
                experience=types.SimpleNamespace(dataset=dataset),
                device=torch.device("cpu"),
                train_mb_size=2,
            )
        )
        with torch.no_grad():
            model.weight[0, 0].add_(0.05)

        strategy = types.SimpleNamespace(
            clock=types.SimpleNamespace(train_exp_counter=1),
            device=torch.device("cpu"),
            model=model,
            loss=torch.tensor(0.0),
        )
        plugin.before_backward(strategy)

        self.assertGreater(strategy.loss.item(), 0.0)
        self.assertEqual(plugin.last_importance_result.rank, 2)

    def test_low_rank_plugin_accepts_four_parameter_config(self):
        model = make_model()
        dataset = make_dataset()
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
        config = IEWCConfig(lambda_=10.0, tau=0.25)
        plugin = LowRankIEWCPlugin(
            config=config,
            importance_kind="ief_low_rank",
            rank=2,
        )

        plugin.after_training_exp(
            types.SimpleNamespace(
                clock=types.SimpleNamespace(train_exp_counter=0),
                model=model,
                _criterion=criterion,
                optimizer=optimizer,
                experience=types.SimpleNamespace(dataset=dataset),
                device=torch.device("cpu"),
                train_mb_size=2,
            )
        )

        self.assertEqual(plugin.ewc_lambda, config.lambda_)
        self.assertEqual(plugin.tau, config.tau)
        self.assertEqual(plugin.last_importance_result.rank, 2)

    def test_low_rank_plus_diagonal_plugin_penalty_matches_manual_formula(self):
        model = make_model()
        dataset = make_dataset()
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
        plugin = LowRankIEWCPlugin(
            ewc_lambda=10.0,
            importance_kind="ief_low_rank_diag",
            rank=1,
            tau=0.25,
        )

        plugin.after_training_exp(
            types.SimpleNamespace(
                clock=types.SimpleNamespace(train_exp_counter=0),
                model=model,
                _criterion=criterion,
                optimizer=optimizer,
                experience=types.SimpleNamespace(dataset=dataset),
                device=torch.device("cpu"),
                train_mb_size=2,
            )
        )
        saved = torch.cat([param.detach().reshape(-1) for param in model.parameters()])
        with torch.no_grad():
            model.weight[0, 0].add_(0.05)
            model.weight[1, 1].sub_(0.02)
        current = torch.cat([param.detach().reshape(-1) for param in model.parameters()])
        delta = current - saved

        strategy = types.SimpleNamespace(
            clock=types.SimpleNamespace(train_exp_counter=1),
            device=torch.device("cpu"),
            model=model,
            loss=torch.tensor(0.0),
        )
        plugin.before_backward(strategy)
        result = plugin.last_importance_result
        low_rank_penalty = (
            result.eigenvalues * (result.eigenvectors @ delta).pow(2)
        ).sum()
        diagonal_penalty = (result.residual_diagonal * delta.pow(2)).sum()
        expected = 10.0 * (low_rank_penalty + diagonal_penalty)

        self.assertTrue(torch.allclose(strategy.loss, expected, atol=1e-8))

    def test_low_rank_plugin_uses_seeded_random_importance_subset(self):
        dataset = TensorDataset(
            torch.arange(20, dtype=torch.float32).reshape(10, 2),
            torch.zeros(10, dtype=torch.long),
            torch.zeros(10, dtype=torch.long),
        )
        plugin = LowRankIEWCPlugin(
            ewc_lambda=1.0,
            rank=1,
            max_importance_samples=4,
            importance_sample_seed=123,
        )

        subset = plugin._importance_subset(dataset, experience_index=2)
        chosen = [int(subset[idx][0][0].item() // 2) for idx in range(len(subset))]
        expected = torch.randperm(
            len(dataset), generator=torch.Generator().manual_seed(125)
        )[:4].tolist()

        self.assertEqual(chosen, expected)

    def test_low_rank_importance_rejects_nonfinite_model_outputs(self):
        model = make_model()
        with torch.no_grad():
            model.weight.fill_(float("nan"))
        dataset = make_dataset()
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

        with self.assertRaises(FloatingPointError):
            LowRankImportanceEstimator(kind="ef_low_rank", rank=1).compute(
                model=model,
                criterion=criterion,
                optimizer=optimizer,
                dataset=dataset,
                device=torch.device("cpu"),
                batch_size=2,
            )


if __name__ == "__main__":
    unittest.main()

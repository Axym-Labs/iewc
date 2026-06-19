import unittest

import torch
from torch import nn
from torch.utils.data import TensorDataset

from iewc.config import IEWCConfig
from iewc.importance import ImportanceEstimator


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


def flatten_importances(importances):
    return torch.cat([value.data.reshape(-1) for value in importances.values()])


class ImportanceEstimatorTests(unittest.TestCase):
    def test_ef_matches_manual_squared_per_sample_gradient_average(self):
        model = make_model()
        dataset = make_dataset()
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

        estimator = ImportanceEstimator(kind="ef")
        result = estimator.compute(
            model=model,
            criterion=criterion,
            optimizer=optimizer,
            dataset=dataset,
            device=torch.device("cpu"),
            batch_size=2,
        )

        manual = torch.zeros_like(model.weight)
        for x, y, _ in dataset:
            optimizer.zero_grad()
            loss = criterion(model(x.unsqueeze(0)), y.unsqueeze(0))
            loss.backward()
            manual += model.weight.grad.detach().pow(2)
        manual /= len(dataset)

        self.assertTrue(
            torch.allclose(flatten_importances(result.importances), manual.reshape(-1))
        )

    def test_ewc_dr_uses_reversed_logits_for_classification_importance(self):
        model = make_model()
        dataset = make_dataset()
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

        ef_result = ImportanceEstimator(kind="ef").compute(
            model=model,
            criterion=criterion,
            optimizer=optimizer,
            dataset=dataset,
            device=torch.device("cpu"),
            batch_size=2,
        )
        dr_result = ImportanceEstimator(kind="ewc_dr").compute(
            model=model,
            criterion=criterion,
            optimizer=optimizer,
            dataset=dataset,
            device=torch.device("cpu"),
            batch_size=2,
        )

        ef = flatten_importances(ef_result.importances)
        dr = flatten_importances(dr_result.importances)

        self.assertGreater(dr.sum().item(), ef.sum().item() * 10.0)

    def test_ief_divides_squared_gradients_by_loss_scale_plus_tau(self):
        model = make_model()
        dataset = make_dataset()
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
        tau = 0.25

        result = ImportanceEstimator(kind="ief_diag", tau=tau).compute(
            model=model,
            criterion=criterion,
            optimizer=optimizer,
            dataset=dataset,
            device=torch.device("cpu"),
            batch_size=2,
        )

        manual = torch.zeros_like(model.weight)
        scales = []
        for x, y, _ in dataset:
            optimizer.zero_grad()
            logits = model(x.unsqueeze(0))
            loss = criterion(logits, y.unsqueeze(0))
            logits_grad = torch.autograd.grad(loss, logits, retain_graph=True)[0]
            scale = logits_grad.detach().pow(2).sum()
            loss.backward()
            manual += model.weight.grad.detach().pow(2) / (scale + tau)
            scales.append(scale.item())
        manual /= len(dataset)

        self.assertTrue(
            torch.allclose(flatten_importances(result.importances), manual.reshape(-1))
        )
        self.assertEqual(result.sample_count, len(dataset))
        self.assertEqual(len(result.loss_scales), len(dataset))
        self.assertTrue(torch.allclose(result.loss_scales, torch.tensor(scales)))

    def test_ief_accepts_four_parameter_config(self):
        model = make_model()
        dataset = make_dataset()
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
        config = IEWCConfig(
            lambda_=5.0,
            tau=0.25,
            geometry="euclidean",
            sample_weighting="uniform",
        )

        configured = ImportanceEstimator(kind="ief_diag", config=config).compute(
            model=model,
            criterion=criterion,
            optimizer=optimizer,
            dataset=dataset,
            device=torch.device("cpu"),
            batch_size=2,
        )
        direct = ImportanceEstimator(kind="ief_diag", tau=0.25).compute(
            model=model,
            criterion=criterion,
            optimizer=optimizer,
            dataset=dataset,
            device=torch.device("cpu"),
            batch_size=2,
        )

        self.assertTrue(
            torch.allclose(
                flatten_importances(configured.importances),
                flatten_importances(direct.importances),
            )
        )

    def test_config_accepts_explicit_weighting_modes_and_rejects_unknown(self):
        for weighting in ("uniform", "gss_residual", "fromp_trace"):
            IEWCConfig(lambda_=1.0, tau=0.0, sample_weighting=weighting).validate()
        config = IEWCConfig(lambda_=1.0, tau=0.0, sample_weighting="not_uniform")
        with self.assertRaises(ValueError):
            config.validate()

    def test_ief_can_use_wasserstein_output_metric_for_loss_scale(self):
        model = nn.Linear(2, 4, bias=False)
        with torch.no_grad():
            model.weight.copy_(
                torch.tensor(
                    [
                        [1.0, 0.0],
                        [0.0, 1.0],
                        [-1.0, 0.0],
                        [0.0, -1.0],
                    ]
                )
            )
        x = torch.tensor([[2.0, 0.0], [-2.0, 0.0]])
        y = torch.tensor([0, 2])
        task = torch.zeros(len(y), dtype=torch.long)
        dataset = TensorDataset(x, y, task)
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

        euclidean = ImportanceEstimator(
            kind="ief_diag", output_metric="euclidean"
        ).compute(
            model=model,
            criterion=criterion,
            optimizer=optimizer,
            dataset=dataset,
            device=torch.device("cpu"),
            batch_size=2,
        )
        wasserstein = ImportanceEstimator(
            kind="ief_diag", output_metric="wasserstein_1d_cdf"
        ).compute(
            model=model,
            criterion=criterion,
            optimizer=optimizer,
            dataset=dataset,
            device=torch.device("cpu"),
            batch_size=2,
        )

        self.assertFalse(
            torch.allclose(euclidean.loss_scales, wasserstein.loss_scales)
        )
        self.assertTrue(torch.all(wasserstein.loss_scales > 0))

if __name__ == "__main__":
    unittest.main()

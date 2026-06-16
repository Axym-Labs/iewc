import types
import unittest

import torch
from torch import nn
from torch.utils.data import TensorDataset

from avalanche.training.utils import copy_params_dict
from iewc.config import IEWCConfig
from iewc.avalanche import IEWCPlugin
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


class IEWCPluginTests(unittest.TestCase):
    def test_plugin_compute_importances_matches_iewc_estimator(self):
        model = make_model()
        dataset = make_dataset()
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
        tau = 0.25

        plugin = IEWCPlugin(ewc_lambda=10.0, importance_kind="ief_diag", tau=tau)
        plugin_importances = plugin.compute_importances(
            model=model,
            criterion=criterion,
            optimizer=optimizer,
            dataset=dataset,
            device=torch.device("cpu"),
            batch_size=2,
        )
        direct_result = ImportanceEstimator(kind="ief_diag", tau=tau).compute(
            model=model,
            criterion=criterion,
            optimizer=optimizer,
            dataset=dataset,
            device=torch.device("cpu"),
            batch_size=2,
        )

        self.assertTrue(
            torch.allclose(
                flatten_importances(plugin_importances),
                flatten_importances(direct_result.importances),
            )
        )
        self.assertEqual(plugin.last_importance_result.sample_count, len(dataset))
        self.assertEqual(len(plugin.last_importance_result.loss_scales), len(dataset))

    def test_plugin_accepts_four_parameter_config(self):
        model = make_model()
        dataset = make_dataset()
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
        config = IEWCConfig(
            lambda_=10.0,
            tau=0.25,
            geometry="euclidean",
            sample_weighting="uniform",
        )

        plugin = IEWCPlugin(config=config, importance_kind="ief_diag")
        plugin_importances = plugin.compute_importances(
            model=model,
            criterion=criterion,
            optimizer=optimizer,
            dataset=dataset,
            device=torch.device("cpu"),
            batch_size=2,
        )
        direct_result = ImportanceEstimator(kind="ief_diag", config=config).compute(
            model=model,
            criterion=criterion,
            optimizer=optimizer,
            dataset=dataset,
            device=torch.device("cpu"),
            batch_size=2,
        )

        self.assertEqual(plugin.ewc_lambda, config.lambda_)
        self.assertTrue(
            torch.allclose(
                flatten_importances(plugin_importances),
                flatten_importances(direct_result.importances),
            )
        )

    def test_plugin_keeps_avalanche_ewc_penalty_path(self):
        model = make_model()
        dataset = make_dataset()
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
        plugin = IEWCPlugin(ewc_lambda=10.0, importance_kind="ewc_dr")

        importances = plugin.compute_importances(
            model=model,
            criterion=criterion,
            optimizer=optimizer,
            dataset=dataset,
            device=torch.device("cpu"),
            batch_size=2,
        )
        plugin.update_importances(importances, 0)
        plugin.saved_params[0] = copy_params_dict(model)

        with torch.no_grad():
            model.weight.add_(0.05)

        strategy = types.SimpleNamespace(
            clock=types.SimpleNamespace(train_exp_counter=1),
            device=torch.device("cpu"),
            model=model,
            loss=torch.tensor(0.0),
        )
        plugin.before_backward(strategy)

        self.assertGreater(strategy.loss.item(), 0.0)


if __name__ == "__main__":
    unittest.main()

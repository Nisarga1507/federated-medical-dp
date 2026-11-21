import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import flwr as fl
from flwr.common import Metrics, ndarrays_to_parameters, parameters_to_ndarrays
import torch
from models.unet_model import MedicalUNet


def weighted_average(metrics: List[Tuple[int, Metrics]]) -> Metrics:
    if not metrics:
        return {}
    total_examples = sum(num_examples for num_examples, _ in metrics)
    aggregated: Dict[str, float] = {}

    numeric_keys = set()
    for _, m in metrics:
        for k, v in m.items():
            if isinstance(v, (int, float)):
                numeric_keys.add(k)

    for key in numeric_keys:
        weighted_sum = sum(
            num_examples * float(m[key])
            for num_examples, m in metrics
            if key in m and isinstance(m[key], (int, float))
        )
        weight_denom = sum(
            num_examples
            for num_examples, m in metrics
            if key in m and isinstance(m[key], (int, float))
        )
        if weight_denom > 0:
            aggregated[key] = weighted_sum / weight_denom

    aggregated["total_examples"] = float(total_examples)
    return aggregated


class SavingFedAvg(fl.server.strategy.FedAvg):
    def __init__(self, cfg: dict, **kwargs):
        super().__init__(**kwargs)
        self.cfg = cfg
        self.output_dir = Path(cfg["artifacts"]["output_dir"])
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.history: List[Dict] = []

    def aggregate_fit(self, server_round, results, failures):
        aggregated_params, metrics = super().aggregate_fit(server_round, results, failures)
        if aggregated_params is not None:
            ndarrays = parameters_to_ndarrays(aggregated_params)
            model = MedicalUNet(
                in_channels=int(self.cfg["data"]["in_channels"]),
                out_channels=int(self.cfg["data"]["out_channels"]),
            )
            keys = list(model.state_dict().keys())
            state_dict = {
                k: torch.tensor(arr, dtype=model.state_dict()[k].dtype)
                for k, arr in zip(keys, ndarrays)
            }
            torch.save(state_dict, self.output_dir / "global_unet.pt")
            self.history.append({
                "round": server_round,
                "fit_metrics": metrics,
            })
            self._write_report()
        return aggregated_params, metrics

    def aggregate_evaluate(self, server_round, results, failures):
        loss, metrics = super().aggregate_evaluate(server_round, results, failures)
        entry = next((item for item in self.history if item["round"] == server_round), None)
        eval_payload = {
            "loss": float(loss) if loss is not None else None,
            "metrics": metrics,
        }
        if entry is not None:
            entry["eval"] = eval_payload
        else:
            self.history.append({"round": server_round, "eval": eval_payload})
        self._write_report()
        return loss, metrics

    def _write_report(self) -> None:
        report = {
            "config": self.cfg,
            "rounds": self.history,
        }
        (self.output_dir / "federated_metrics.json").write_text(json.dumps(report, indent=2))


def run_federated_server(cfg: dict) -> None:
    model = MedicalUNet(
        in_channels=int(cfg["data"]["in_channels"]),
        out_channels=int(cfg["data"]["out_channels"]),
    )
    init_ndarrays = [val.detach().cpu().numpy() for _, val in model.state_dict().items()]
    initial_parameters = ndarrays_to_parameters(init_ndarrays)

    min_clients = int(cfg["federated"]["min_clients"])
    strategy = SavingFedAvg(
        cfg=cfg,
        fraction_fit=1.0,
        fraction_evaluate=1.0,
        min_fit_clients=min_clients,
        min_evaluate_clients=min_clients,
        min_available_clients=min_clients,
        initial_parameters=initial_parameters,
        fit_metrics_aggregation_fn=weighted_average,
        evaluate_metrics_aggregation_fn=weighted_average,
    )

    fl.server.start_server(
        server_address=str(cfg["federated"]["server_address"]),
        config=fl.server.ServerConfig(num_rounds=int(cfg["federated"]["num_rounds"])),
        strategy=strategy,
    )

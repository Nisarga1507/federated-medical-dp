import argparse
from pathlib import Path
import flwr as fl
import yaml
from client.client_node import HospitalClient


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a federated hospital client node.")
    parser.add_argument("--client-id", type=int, required=True, help="Hospital client ID (e.g. 1 or 2)")
    parser.add_argument("--config", type=str, default="configs/default_config.yaml")
    args = parser.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text())
    client = HospitalClient(client_id=args.client_id, cfg=cfg).to_client()

    fl.client.start_client(
        server_address=str(cfg["federated"]["server_address"]),
        client=client,
    )


if __name__ == "__main__":
    main()

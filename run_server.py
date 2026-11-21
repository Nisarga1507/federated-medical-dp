import argparse
from pathlib import Path
import yaml
from server.aggregation_server import run_federated_server


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the federated aggregation server.")
    parser.add_argument("--config", type=str, default="configs/default_config.yaml")
    args = parser.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text())
    run_federated_server(cfg)


if __name__ == "__main__":
    main()

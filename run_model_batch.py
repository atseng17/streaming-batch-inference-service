import argparse
import time
from typing import Any, Dict, List

import torch

# Ensure the class definition exists in the import graph for torch.load
from create_model import InefficientModel  # noqa: F401


def build_sample_events() -> List[Dict[str, Any]]:
    now = int(time.time())
    return [
        {"user_id": "user-1", "timestamp": now, "features": [0.1, 0.2, 0.3]},
        {"user_id": "user-1", "timestamp": now, "features": [0.2, 0.1, 0.0]},
        {"user_id": "user-2", "timestamp": now, "features": [0.9, 0.8, 0.7]},
    ]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", default="inefficient_model.pt")
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    model = torch.load(args.model_path, map_location=args.device)
    model.eval()

    events = build_sample_events()
    batch_features = [e["features"] for e in events]

    x = torch.tensor(batch_features, dtype=torch.float32, device=args.device)

    with torch.no_grad():
        y = model(x)

    if isinstance(y, torch.Tensor):
        preds = y.detach().cpu().view(-1).tolist()
    else:
        preds = list(y)

    for event, pred in zip(events, preds):
        print(
            f"user_id={event['user_id']} timestamp={event['timestamp']} features={event['features']} prediction={pred}"
        )


if __name__ == "__main__":
    main()

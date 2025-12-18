import asyncio
import random
import sys
import time

import cProfile as cprofile_module
import torch

import app
from create_model import InefficientModel

setattr(cprofile_module, "InefficientModel", InefficientModel)
setattr(sys.modules["__main__"], "InefficientModel", InefficientModel)


async def main(batches: int = 5000, batch_size: int = 32, feature_dim: int = 3, users: int = 50) -> None:
    class _App:
        pass

    a = _App()
    a.state = _App()

    a.state.model = torch.load("inefficient_model.pt")
    a.state.model.eval()

    a.state.stats = {
        "request_count": 0,
        "inference_count": 0,
        "avg_latency": 0,
        "total_latency": 0,
        "batch_count": 0,
        "avg_batch_size": 0,
        "total_batch_size": 0,
        "median_of_medians": None,
    }
    a.state.user_data = {}

    for _ in range(batches):
        batch = [[random.random() for _ in range(feature_dim)] for __ in range(batch_size)]
        preds = await app.run_batched_inference(a, batch)

        now = int(time.time())
        for i, p in enumerate(preds):
            app.update_rolling_median(a, f"u{i % users}", now - i, p)

    app.calculate_median_of_medians(a)


if __name__ == "__main__":
    asyncio.run(main())

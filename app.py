import asyncio
import logging
import time
import uvicorn
from fastapi import FastAPI, HTTPException, BackgroundTasks, Request
from contextlib import asynccontextmanager
from pydantic import BaseModel
from typing import List, Dict, Optional, Any
import torch
import statistics

# Import the model class to ensure it's available when loading the model
from create_model import InefficientModel

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Application state
def _fresh_stats():
    return {
        "request_count": 0,
        "inference_count": 0,
        "avg_latency": 0,
        "total_latency": 0,
        "batch_count": 0,
        "avg_batch_size": 0,
        "total_batch_size": 0,
        "median_of_medians": None,
    }

app_state = {
    "user_data": {},  # Dictionary to store user data and rolling medians
    "stats": _fresh_stats()
}


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifecycle manager"""
    # Load model during app initalization
    logger.info("Application startup: Loading model...")
    try:
        app.state.model = torch.load("inefficient_model.pt")
        app.state.model.eval()
        app.state.user_data = app_state["user_data"]
        app.state.stats = app_state["stats"]
        logger.info("Model loaded successfully")
    except Exception as e:
        logger.exception("Error loading model")
        raise

    # TODO: in Next Steps, connect to redis here

    app.state.batch_queue = asyncio.Queue()

    # Start the batch processor
    app.state.batch_processor_task = asyncio.create_task(batch_processor(app))
    logger.info("Service started successfully with batch processing")
    
    yield
    
    # Clean up resources
    logger.info("Application shutdown: Cleaning up resources...")
    if hasattr(app.state, 'batch_processor_task'):
        app.state.batch_processor_task.cancel()
        try:
            await app.state.batch_processor_task
        except asyncio.CancelledError:
            pass
    logger.info("Resources released.")


app = FastAPI(title="ML Prediction Service", lifespan=lifespan)

# Batch processing variables
batch_size = 32  # Maximum batch size for inference

# Define data models
class Feature(BaseModel):
    user_id: str
    timestamp: int
    features: List[float]

class EventBatch(BaseModel):
    events: List[Feature]

class Prediction(BaseModel):
    user_id: str
    timestamp: int
    features: List[float]
    prediction: float

class Stats(BaseModel):
    request_count: int
    inference_count: int
    avg_latency: float
    batch_count: int
    avg_batch_size: float
    median_of_medians: Optional[float] = None

# Function to run inference on a single sample
def run_inference(request: Request, features):
    model = request.app.state.model
    stats = request.app.state.stats
    
    # Convert features to tensor
    tensor_features = torch.tensor([features], dtype=torch.float32)
    
    # Run inference
    with torch.no_grad():
        start_time = time.time()
        prediction = model(tensor_features).item()
        inference_time = time.time() - start_time
        
        # Update stats
        stats["inference_count"] += 1
        stats["total_latency"] += inference_time
        stats["avg_latency"] = stats["total_latency"] / stats["inference_count"]
            
    return prediction

# Function to run batched inference
async def run_batched_inference(app, batch_features):
    model = app.state.model
    stats = app.state.stats
    
    # Convert features to tensor
    tensor_features = torch.tensor(batch_features, dtype=torch.float32)

    # add sleep to simulateFor testing when high cpu load with high rps
    # time.sleep(0.1)
    
    # Run inference
    with torch.no_grad():
        start_time = time.time()
        predictions = model(tensor_features).tolist()
        inference_time = time.time() - start_time
        
        # Update stats
        batch_size = len(batch_features)
        stats["inference_count"] += batch_size
        stats["total_latency"] += inference_time
        stats["avg_latency"] = stats["total_latency"] / stats["inference_count"]
        stats["batch_count"] += 1
        stats["total_batch_size"] += batch_size
        stats["avg_batch_size"] = stats["total_batch_size"] / stats["batch_count"]
            
    return predictions

# Batch processor task (opportunistic batching)
async def batch_processor(app):
    batch_queue = app.state.batch_queue
    while True:
        # Collect items for the batch
        batch_items = []
        batch_features = []
        
        try:
            # Get the first item
            item = await batch_queue.get()
            batch_items.append(item)
            batch_features.append(item["features"])
            batch_queue.task_done()
            
            # Try to get more items up to batch_size
            for _ in range(batch_size - 1):
                try:
                    item = batch_queue.get_nowait()
                    batch_items.append(item)
                    batch_features.append(item["features"])
                    batch_queue.task_done()
                except asyncio.QueueEmpty:
                    break
            
            # Process the batch
            if batch_items:
                # Run inference
                # logger.info("batch size: %d", len(batch_features))
                predictions = await run_batched_inference(app, batch_features)
                
                # Update rolling medians
                for i, item in enumerate(batch_items):
                    update_rolling_median(app, item["user_id"], item["timestamp"], predictions[i])
            # logger.info(f"user data: {app.state.user_data}")
            # logger.info(f"stats: {app.state.stats}")
        except Exception as e:
            logger.exception("Error in batch processing")
            await asyncio.sleep(0.1)  # Avoid tight loop in case of errors

# Function to update rolling median with support for out-of-order events, might need another look, to see if out-of-order events are only subjected to calculating medians
def update_rolling_median(app, user_id, timestamp, prediction):
    user_data = app.state.user_data
    current_time = int(time.time())
    five_min_ago = current_time - 300  # 5 minutes = 300 seconds
    
    if user_id not in user_data:
        user_data[user_id] = []
    
    # Add new prediction with timestamp
    user_data[user_id].append((timestamp, prediction))
    
    # Sort by timestamp to handle out-of-order events
    user_data[user_id].sort(key=lambda x: x[0])
    
    # Remove predictions older than 5 minutes
    while user_data[user_id] and user_data[user_id][0][0] < five_min_ago:
        user_data[user_id].pop(0)

# Function to get rolling median for a user
def get_user_median(app, user_id):
    user_data = app.state.user_data
    if user_id not in user_data or not user_data[user_id]:
        return None
    
    # Extract predictions from the list
    predictions = [p[1] for p in user_data[user_id]]
    
    # Calculate median
    return statistics.median(predictions) if predictions else None

# Function to calculate median of medians
def calculate_median_of_medians(app):
    user_data = app.state.user_data
    stats = app.state.stats
    # Get all user medians
    medians = []
    for user_id in user_data:
        user_median = get_user_median(app, user_id)
        if user_median is not None:
            medians.append(user_median)
    # This is part of the stretch goal
    if medians:
        stats["median_of_medians"] = statistics.median(medians)
        return stats["median_of_medians"]
    return None

# API endpoints
@app.post("/ingest")
async def ingest_events(request: Request, event_batch: EventBatch, background_tasks: BackgroundTasks):
    stats = request.app.state.stats
    stats["request_count"] += 1
    
    # Add events to the batch queue
    for event in event_batch.events:
        await request.app.state.batch_queue.put({
            "user_id": event.user_id,
            "timestamp": event.timestamp,
            "features": event.features
        })
    
    # Recompute on every ingest request
    background_tasks.add_task(calculate_median_of_medians, request.app)
    
    return {"queued": len(event_batch.events)}

@app.get("/users/{user_id}/median")
async def get_median(user_id: str, request: Request):
    stats = request.app.state.stats
    stats["request_count"] += 1
    
    median = get_user_median(request.app, user_id)
    
    if median is None:
        raise HTTPException(status_code=404, detail=f"No data found for user {user_id}")
    
    return {"user_id": user_id, "median": median}

@app.get("/users/{user_id}/history")
async def get_history(user_id: str, request: Request):
    stats = request.app.state.stats
    stats["request_count"] += 1

    user_data = request.app.state.user_data
    if user_id not in user_data or not user_data[user_id]:
        raise HTTPException(status_code=404, detail=f"No data found for user {user_id}")

    history = [
        {"timestamp": ts, "prediction": pred}
        for ts, pred in user_data[user_id]
    ]

    return {"user_id": user_id, "history": history}

@app.get("/stats")
async def get_stats(request: Request):
    stats = request.app.state.stats
    stats["request_count"] += 1
    
    return Stats(
        request_count=stats["request_count"], # is this needed?
        inference_count=stats["inference_count"], # is this needed?
        avg_latency=stats["avg_latency"], # is this needed?
        batch_count=stats["batch_count"], # is this needed?
        avg_batch_size=stats["avg_batch_size"], # is this needed?
        median_of_medians=stats["median_of_medians"] # stretch goal
    )


@app.post("/reset")
async def reset_state(request: Request):
    request.app.state.user_data.clear()

    stats = request.app.state.stats
    stats.clear()
    stats.update(_fresh_stats())

    if hasattr(request.app.state, "batch_queue"):
        while True:
            try:
                request.app.state.batch_queue.get_nowait()
                request.app.state.batch_queue.task_done()
            except asyncio.QueueEmpty:
                break

    return {"status": "ok"}


# Main function
if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)

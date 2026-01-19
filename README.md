# Streaming Batch Inference Service

A lightweight service that processes a real-time stream of JSON events and serves ML-powered predictions concurrently.

## Features

- Data ingestion from a stream of JSON events
- Real-time ML model inference
- Rolling median calculation over 5-minute windows
- HTTP API for querying user medians and service statistics
- Concurrent request handling (ingestion and inference)
- Opportunistic batching (stretch goal)
- Support for out-of-order events in the stream (stretch goal)
- Median of medians calculation in stats endpoint (stretch goal)
- Profiling (stretch goal)

## Requirements

- Python 3.11+
- PyTorch
- FastAPI
- Docker

Check `requirements.txt` for the list of dependencies.

## Setup

### Local Development

1. Create a virtual environment:
   ```
   python -m venv .env
   ```

2. Activate the virtual environment:
   ```
   source .env/bin/activate
   ```

3. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

4. Create the ML model:
   ```
   python create_model.py
   ```

5. Run the service:
   ```
   python app.py
   ```

### Docker Deployment

1. Build the Docker image:
   ```
   docker build -t batch-processing-service .
   ```
   For macbook users (arm64):
   ```
   docker build --platform=linux/amd64 -t batch-processing-service .
   ```

2. Run the Docker container:
   ```
   docker run -p 8000:8000 batch-processing-service
   ```

## Usage

### API Endpoints

- **POST /ingest**: Ingest a batch of events
  ```
  curl -X POST http://localhost:8000/ingest \
  -H "Content-Type: application/json" \
  -d '{"events":[{"user_id":"user-1","timestamp": '"$(date +%s)"', "features":[0.1,0.2,0.3]},{"user_id":"user-1","timestamp": '"$(date +%s)"', "features":[0.2,0.1,0.0]},{"user_id":"user-2","timestamp": '"$(date +%s)"', "features":[0.9,0.8,0.7]}]}'
  ```

- **GET /users/{user_id}/median**: Get the rolling median for a specific user
  ```
  curl http://localhost:8000/users/user-1/median
  ```

- **GET /users/{user_id}/history**: Get the full stored history for a specific user
  ```
  curl http://localhost:8000/users/user-1/history
  ```

- **GET /stats**: Get service statistics
  ```
  curl http://localhost:8000/stats
  ```

- **POST /reset**: Reset in-memory stats and user rolling medians
  ```
  curl -X POST http://localhost:8000/reset
  ```

### Generate Events

Use the provided event generator to send events to the service:

```
python event_generator.py
```

By default, this sends events to `http://localhost:8000/ingest`.

## Testing
### Unit tests
```
pytest -xvs tests
```

### Test end to end
Start the service locally (`python app.py`) or via Docker (`docker run -p 8000:8000 batch-processing-service`).

```Terminal 2
python -c "import event_generator; event_generator.run(rps=10, duration_sec=1, users=5)"
```

```Terminal 3
curl http://localhost:8000/stats
curl http://localhost:8000/users/user-1/median
curl http://localhost:8000/users/user-10/median
curl http://localhost:8000/users/user-25/median
```

### Profiling
```
python -m cProfile -o prof.out profile_harness.py

python -c "import pstats; p=pstats.Stats('prof.out'); p.strip_dirs().sort_stats('cumtime'); p.print_stats('app.py|profile_harness.py|create_model.py', 50)"
```
The following results are returned, showing the sorting part of the code is the bottleneck. The original purpose of sorting is to deal with out of order events.
```
   ncalls  tottime  percall  cumtime  percall filename:lineno(function)
        1    0.000    0.000   53.616   53.616 profile_harness.py:1(<module>)
        1    0.099    0.099   51.721   51.721 profile_harness.py:16(main)
   160000    0.154    0.000   50.929    0.000 app.py:204(update_user_history)
400080000   19.068    0.000   19.068    0.000 app.py:222(<lambda>)
     5000    0.033    0.000    0.550    0.000 app.py:121(run_batched_inference)
        1    0.000    0.000    0.512    0.512 app.py:1(<module>)
     5000    0.049    0.000    0.384    0.000 create_model.py:16(forward)
     5000    0.038    0.000    0.118    0.000 profile_harness.py:39(<listcomp>)
        1    0.000    0.000    0.020    0.020 app.py:241(calculate_median_of_medians)
       32    0.000    0.000    0.019    0.001 app.py:229(get_user_median)
       32    0.006    0.000    0.006    0.000 app.py:235(<listcomp>)
        1    0.000    0.000    0.001    0.001 create_model.py:1(<module>)
        1    0.000    0.000    0.000    0.000 app.py:89(Feature)
        1    0.000    0.000    0.000    0.000 app.py:94(EventBatch)
        1    0.000    0.000    0.000    0.000 app.py:103(Stats)
        1    0.000    0.000    0.000    0.000 app.py:98(IngestResponse)
        1    0.000    0.000    0.000    0.000 app.py:114(UserMedianResponse)
        1    0.000    0.000    0.000    0.000 app.py:19(_fresh_stats)
        1    0.000    0.000    0.000    0.000 create_model.py:5(InefficientModel)
        1    0.000    0.000    0.000    0.000 profile_harness.py:17(_App)
```


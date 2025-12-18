import pytest
import asyncio
import time
from unittest.mock import MagicMock, patch, AsyncMock

# Import the batch processor function
from app import batch_processor


@pytest.mark.asyncio
class TestBatchProcessor:
    """Tests for the batch processor"""
    
    async def test_batch_processor_processes_single_item(self):
        """Test that batch processor processes a single item correctly"""
        # Mock app and dependencies
        app = MagicMock()
        app.state.batch_queue = asyncio.Queue()
        app.state.model = MagicMock()
        app.state.model.return_value = [0.5]
        app.state.stats = {
            "inference_count": 0,
            "total_latency": 0,
            "avg_latency": 0,
            "batch_count": 0,
            "total_batch_size": 0,
            "avg_batch_size": 0
        }
        app.state.user_data = {}
        
        # Mock run_batched_inference
        async def mock_run_inference(app, features):
            return [0.5]
        
        # Create a test item
        test_item = {
            "user_id": "user-1",
            "timestamp": int(time.time()),
            "features": [0.1, 0.2, 0.3]
        }
        
        # Clear the queue and add the test item
        batch_queue = app.state.batch_queue
        while not batch_queue.empty():
            try:
                batch_queue.get_nowait()
                batch_queue.task_done()
            except asyncio.QueueEmpty:
                break
        
        await batch_queue.put(test_item)
        
        # Run batch processor for a short time
        with patch('app.run_batched_inference', new=mock_run_inference):
            # Create a task for the batch processor
            processor_task = asyncio.create_task(batch_processor(app))
            
            # Give it a short time to run
            await asyncio.sleep(0.1)
            
            # Cancel the task
            processor_task.cancel()
            try:
                await processor_task
            except asyncio.CancelledError:
                pass
        
        # Check that the item was processed
        assert "user-1" in app.state.user_data
    
    async def test_batch_processor_processes_multiple_items(self):
        """Test that batch processor processes multiple items correctly"""
        # Mock app and dependencies
        app = MagicMock()
        app.state.batch_queue = asyncio.Queue()
        app.state.model = MagicMock()
        app.state.stats = {
            "inference_count": 0,
            "total_latency": 0,
            "avg_latency": 0,
            "batch_count": 0,
            "total_batch_size": 0,
            "avg_batch_size": 0
        }
        app.state.user_data = {}
        
        # Mock run_batched_inference
        async def mock_run_inference(app, features):
            return [0.1, 0.2, 0.3]
        
        # Create test items
        test_items = [
            {
                "user_id": "user-1",
                "timestamp": int(time.time()),
                "features": [0.1, 0.2, 0.3]
            },
            {
                "user_id": "user-2",
                "timestamp": int(time.time()),
                "features": [0.4, 0.5, 0.6]
            },
            {
                "user_id": "user-3",
                "timestamp": int(time.time()),
                "features": [0.7, 0.8, 0.9]
            }
        ]
        
        # Clear the queue and add the test items
        batch_queue = app.state.batch_queue
        while not batch_queue.empty():
            try:
                batch_queue.get_nowait()
                batch_queue.task_done()
            except asyncio.QueueEmpty:
                break
        
        for item in test_items:
            await batch_queue.put(item)
        
        # Run batch processor for a short time
        with patch('app.run_batched_inference', new=mock_run_inference):
            # Create a task for the batch processor
            processor_task = asyncio.create_task(batch_processor(app))
            
            # Give it a short time to run
            await asyncio.sleep(0.1)
            
            # Cancel the task
            processor_task.cancel()
            try:
                await processor_task
            except asyncio.CancelledError:
                pass
        
        # Check that the items were processed
        assert "user-1" in app.state.user_data
        assert "user-2" in app.state.user_data
        assert "user-3" in app.state.user_data
    
    async def test_batch_processor_handles_errors(self):
        """Test that batch processor handles errors gracefully"""
        # Mock app
        app = MagicMock()
        app.state.batch_queue = asyncio.Queue()
        
        # Mock run_batched_inference to raise an exception
        async def mock_run_inference_error(app, features):
            raise ValueError("Test error")
        
        # Create a test item
        test_item = {
            "user_id": "user-1",
            "timestamp": int(time.time()),
            "features": [0.1, 0.2, 0.3]
        }
        
        # Clear the queue and add the test item
        batch_queue = app.state.batch_queue
        while not batch_queue.empty():
            try:
                batch_queue.get_nowait()
                batch_queue.task_done()
            except asyncio.QueueEmpty:
                break
        
        await batch_queue.put(test_item)
        
        # Run batch processor for a short time
        with patch('app.run_batched_inference', new=mock_run_inference_error):
            # Create a task for the batch processor
            processor_task = asyncio.create_task(batch_processor(app))
            
            # Give it a short time to run
            await asyncio.sleep(0.1)
            
            # Cancel the task
            processor_task.cancel()
            try:
                await processor_task
            except asyncio.CancelledError:
                pass
        
        # The test passes if no unhandled exceptions were raised


if __name__ == "__main__":
    pytest.main(["-xvs", "test_batch_processor.py"])

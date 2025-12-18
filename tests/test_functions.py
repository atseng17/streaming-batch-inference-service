import pytest
import time
import torch
import statistics
import numpy as np
from unittest.mock import MagicMock, patch

# Import the functions to test
from app import update_rolling_median, get_user_median, calculate_median_of_medians, run_inference, run_batched_inference


class TestRollingMedian:
    """Tests for the rolling median calculation functions"""

    def test_update_rolling_median(self):
        """Test that update_rolling_median correctly adds and sorts data"""
        # Mock app state
        app = MagicMock()
        app.state.user_data = {}
        
        # Add some predictions
        user_id = "user-1"
        current_time = int(time.time())
        
        # Add predictions in random order
        update_rolling_median(app, user_id, current_time - 10, 0.5)
        update_rolling_median(app, user_id, current_time - 30, 0.3)
        update_rolling_median(app, user_id, current_time - 20, 0.4)
        
        # Check that data was added and sorted by timestamp
        assert len(app.state.user_data[user_id]) == 3
        assert app.state.user_data[user_id][0][0] == current_time - 30
        assert app.state.user_data[user_id][1][0] == current_time - 20
        assert app.state.user_data[user_id][2][0] == current_time - 10
        
        # Check prediction values
        assert app.state.user_data[user_id][0][1] == 0.3
        assert app.state.user_data[user_id][1][1] == 0.4
        assert app.state.user_data[user_id][2][1] == 0.5

    def test_update_rolling_median_removes_old_data(self):
        """Test that update_rolling_median removes data older than 5 minutes"""
        # Mock app state
        app = MagicMock()
        app.state.user_data = {}
        
        user_id = "user-1"
        current_time = int(time.time())
        
        # Add old prediction (more than 5 minutes old)
        update_rolling_median(app, user_id, current_time - 301, 0.1)
        
        # Add recent predictions
        update_rolling_median(app, user_id, current_time - 10, 0.5)
        update_rolling_median(app, user_id, current_time - 30, 0.3)
        
        # Check that old data was removed
        assert len(app.state.user_data[user_id]) == 2
        assert app.state.user_data[user_id][0][0] == current_time - 30
        assert app.state.user_data[user_id][1][0] == current_time - 10

    def test_get_user_median(self):
        """Test that get_user_median correctly calculates the median"""
        # Mock app state
        app = MagicMock()
        app.state.user_data = {
            "user-1": [(1000, 0.1), (1001, 0.5), (1002, 0.3)]
        }
        
        # Calculate median
        median = get_user_median(app, "user-1")
        
        # Check median calculation
        assert median == 0.3
        
        # Test non-existent user
        assert get_user_median(app, "non-existent-user") is None
        
        # Test empty data
        app.state.user_data["empty-user"] = []
        assert get_user_median(app, "empty-user") is None

    def test_calculate_median_of_medians(self):
        """Test that calculate_median_of_medians correctly calculates the median of medians"""
        # Mock app state
        app = MagicMock()
        app.state.user_data = {
            "user-1": [(1000, 0.1), (1001, 0.5), (1002, 0.3)],  # median = 0.3
            "user-2": [(1000, 0.2), (1001, 0.4), (1002, 0.6)],  # median = 0.4
            "user-3": [(1000, 0.7), (1001, 0.9), (1002, 0.5)]   # median = 0.7
        }
        app.state.stats = {"median_of_medians": None}
        
        # Calculate median of medians
        result = calculate_median_of_medians(app)
        
        # Check median of medians calculation
        assert result == 0.4
        assert app.state.stats["median_of_medians"] == 0.4


class TestInference:
    """Tests for the inference functions"""
    
    @patch('torch.load')
    def test_run_inference(self, mock_load):
        """Test that run_inference correctly runs inference on a single sample"""
        # Mock model and request
        mock_model = MagicMock()
        mock_model.return_value = torch.tensor([0.5])
        
        request = MagicMock()
        request.app.state.model = mock_model
        request.app.state.stats = {
            "inference_count": 0,
            "total_latency": 0,
            "avg_latency": 0
        }
        
        # Run inference
        features = [0.1, 0.2, 0.3]
        result = run_inference(request, features)
        
        # Check result
        assert result == 0.5
        
        # Check stats were updated
        assert request.app.state.stats["inference_count"] == 1
        assert request.app.state.stats["total_latency"] > 0
        assert request.app.state.stats["avg_latency"] > 0

    @pytest.mark.asyncio
    async def test_run_batched_inference(self): # TODO: need to further lookinto this
        """Test that run_batched_inference correctly runs inference on a batch"""
        # Mock model and app
        mock_model = MagicMock()
        mock_model.return_value = torch.tensor([0.5, 0.6, 0.7])
        
        app = MagicMock()
        app.state.model = mock_model
        app.state.stats = {
            "inference_count": 0,
            "total_latency": 0,
            "avg_latency": 0,
            "batch_count": 0,
            "total_batch_size": 0,
            "avg_batch_size": 0
        }
        
        # Run batched inference
        batch_features = [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6], [0.7, 0.8, 0.9]]
        results = await run_batched_inference(app, batch_features)
        
        # Check results - using numpy.isclose for floating point comparison
        import numpy as np
        assert len(results) == 3
        assert np.isclose(results[0], 0.5)
        assert np.isclose(results[1], 0.6)
        assert np.isclose(results[2], 0.7)
        
        # Check stats were updated
        assert app.state.stats["inference_count"] == 3
        assert app.state.stats["total_latency"] > 0
        assert app.state.stats["avg_latency"] > 0
        assert app.state.stats["batch_count"] == 1
        assert app.state.stats["total_batch_size"] == 3
        assert app.state.stats["avg_batch_size"] == 3


class TestEndToEnd:
    """End-to-end tests for the full data flow"""
    
    @pytest.mark.asyncio
    async def test_data_flow(self):
        """Test the full data flow from ingestion to median calculation"""
        # Mock app state
        app = MagicMock()
        app.state.user_data = {}
        app.state.stats = {
            "inference_count": 0,
            "total_latency": 0,
            "avg_latency": 0,
            "batch_count": 0,
            "total_batch_size": 0,
            "avg_batch_size": 0,
            "median_of_medians": None
        }
        
        # Mock model that returns predictable values
        mock_model = MagicMock()
        mock_model.return_value = torch.tensor([0.1, 0.2, 0.3, 0.4, 0.5])
        app.state.model = mock_model
        
        # Create batch of events
        batch_features = [
            [0.1, 0.2, 0.3],  # Will return 0.1
            [0.4, 0.5, 0.6],  # Will return 0.2
            [0.7, 0.8, 0.9],  # Will return 0.3
            [0.2, 0.3, 0.4],  # Will return 0.4
            [0.5, 0.6, 0.7]   # Will return 0.5
        ]
        
        # Run batched inference
        predictions = await run_batched_inference(app, batch_features)
        
        # Update rolling medians for different users
        current_time = int(time.time())
        update_rolling_median(app, "user-1", current_time - 10, predictions[0])
        update_rolling_median(app, "user-1", current_time - 20, predictions[1])
        update_rolling_median(app, "user-2", current_time - 10, predictions[2])
        update_rolling_median(app, "user-2", current_time - 20, predictions[3])
        update_rolling_median(app, "user-3", current_time - 10, predictions[4])
        
        # Check user medians - using numpy.isclose for floating point comparison
        import numpy as np
        assert np.isclose(get_user_median(app, "user-1"), 0.15)  # median of [0.1, 0.2]
        assert np.isclose(get_user_median(app, "user-2"), 0.35)  # median of [0.3, 0.4]
        assert np.isclose(get_user_median(app, "user-3"), 0.5)   # median of [0.5]
        
        # Calculate median of medians
        median_of_medians = calculate_median_of_medians(app)
        
        # Check median of medians
        assert np.isclose(median_of_medians, 0.35)  # median of [0.15, 0.35, 0.5]
        assert np.isclose(app.state.stats["median_of_medians"], 0.35)


if __name__ == "__main__":
    pytest.main(["-xvs", "test_functions.py"])

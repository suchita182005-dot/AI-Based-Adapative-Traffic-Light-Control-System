"""
Traffic Light Controller Module for AI-Based Adaptive Traffic Light Control System
This module implements the core logic for adaptive traffic light control based on
real-time vehicle density data.
"""

import time
import threading
import logging
from typing import Dict, List, Tuple
from enum import Enum
from dataclasses import dataclass
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
import pickle
import os

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class LightState(Enum):
    """Enumeration for traffic light states."""
    RED = "RED"
    YELLOW = "YELLOW"
    GREEN = "GREEN"

class Direction(Enum):
    """Enumeration for traffic directions."""
    NORTH_SOUTH = "NORTH_SOUTH"
    EAST_WEST = "EAST_WEST"

@dataclass
class TrafficData:
    """Data class for storing traffic information."""
    direction: Direction
    vehicle_count: int
    density: float
    timestamp: float
    emergency_vehicle: bool = False

@dataclass
class LightTiming:
    """Data class for storing light timing information."""
    green_duration: int
    yellow_duration: int = 3
    red_duration: int = 1

class AdaptiveTrafficController:
    """
    Main controller class for the adaptive traffic light system.
    Uses machine learning to optimize traffic light timing based on real-time data.
    """
    
    def __init__(self, intersection_id: str = "INT_001"):
        """
        Initialize the adaptive traffic controller.
        
        Args:
            intersection_id: Unique identifier for the intersection
        """
        self.intersection_id = intersection_id
        self.current_state = {
            Direction.NORTH_SOUTH: LightState.RED,
            Direction.EAST_WEST: LightState.GREEN
        }
        
        # Default timing parameters
        self.min_green_time = 10  # Minimum green light duration (seconds)
        self.max_green_time = 60  # Maximum green light duration (seconds)
        self.default_green_time = 30  # Default green light duration (seconds)
        self.yellow_time = 3  # Yellow light duration (seconds)
        self.red_clearance_time = 1  # All-red clearance time (seconds)
        
        # Traffic data storage
        self.traffic_history: List[TrafficData] = []
        self.current_traffic_data = {
            Direction.NORTH_SOUTH: TrafficData(Direction.NORTH_SOUTH, 0, 0.0, time.time()),
            Direction.EAST_WEST: TrafficData(Direction.EAST_WEST, 0, 0.0, time.time())
        }
        
        # Machine learning components
        self.ml_model = LinearRegression()
        self.scaler = StandardScaler()
        self.model_trained = False
        
        # Control flags
        self.running = False
        self.emergency_mode = False
        self.control_thread = None
        
        # Performance metrics
        self.cycle_count = 0
        self.total_wait_time = 0
        self.efficiency_score = 0.0
        
    def update_traffic_data(self, direction: Direction, vehicle_count: int, density: float, emergency_vehicle: bool = False):
        """
        Update traffic data for a specific direction.
        
        Args:
            direction: Traffic direction
            vehicle_count: Number of vehicles detected
            density: Traffic density value
            emergency_vehicle: Whether an emergency vehicle is detected
        """
        self.current_traffic_data[direction] = TrafficData(
            direction=direction,
            vehicle_count=vehicle_count,
            density=density,
            timestamp=time.time(),
            emergency_vehicle=emergency_vehicle
        )
        
        # Add to history for machine learning
        self.traffic_history.append(self.current_traffic_data[direction])
        
        # Keep only recent history (last 1000 entries)
        if len(self.traffic_history) > 1000:
            self.traffic_history = self.traffic_history[-1000:]
        
        # Check for emergency vehicles
        if emergency_vehicle:
            self.handle_emergency_vehicle(direction)
        
        logger.info(f"Updated traffic data for {direction.value}: {vehicle_count} vehicles, density: {density:.2f}")
    
    def calculate_adaptive_timing(self, direction: Direction) -> LightTiming:
        """
        Calculate adaptive timing for traffic lights based on current traffic conditions.
        
        Args:
            direction: Traffic direction to calculate timing for
            
        Returns:
            LightTiming object with calculated durations
        """
        current_data = self.current_traffic_data[direction]
        opposite_direction = Direction.EAST_WEST if direction == Direction.NORTH_SOUTH else Direction.NORTH_SOUTH
        opposite_data = self.current_traffic_data[opposite_direction]
        
        # Base calculation using traffic density ratio
        if opposite_data.density > 0:
            density_ratio = current_data.density / (current_data.density + opposite_data.density)
        else:
            density_ratio = 1.0 if current_data.density > 0 else 0.5
        
        # Calculate green time based on density ratio
        green_duration = int(self.min_green_time + 
                           (self.max_green_time - self.min_green_time) * density_ratio)
        
        # Apply machine learning adjustment if model is trained
        if self.model_trained:
            try:
                features = self._extract_features(current_data, opposite_data)
                features_scaled = self.scaler.transform([features])
                ml_adjustment = self.ml_model.predict(features_scaled)[0]
                green_duration = max(self.min_green_time, 
                                   min(self.max_green_time, 
                                       int(green_duration + ml_adjustment)))
            except Exception as e:
                logger.warning(f"ML prediction failed: {e}")
        
        # Emergency vehicle override
        if current_data.emergency_vehicle:
            green_duration = max(green_duration, 45)  # Ensure sufficient time for emergency vehicles
        
        return LightTiming(green_duration=green_duration, yellow_duration=self.yellow_time)
    
    def _extract_features(self, current_data: TrafficData, opposite_data: TrafficData) -> List[float]:
        """
        Extract features for machine learning model.
        
        Args:
            current_data: Traffic data for current direction
            opposite_data: Traffic data for opposite direction
            
        Returns:
            List of feature values
        """
        # Time-based features
        current_time = time.time()
        hour_of_day = (current_time % 86400) / 3600  # Hour of day (0-24)
        day_of_week = (current_time // 86400) % 7  # Day of week (0-6)
        
        # Traffic features
        features = [
            current_data.vehicle_count,
            current_data.density,
            opposite_data.vehicle_count,
            opposite_data.density,
            current_data.vehicle_count - opposite_data.vehicle_count,  # Difference
            current_data.density / (opposite_data.density + 0.1),  # Ratio
            hour_of_day,
            day_of_week,
            1.0 if current_data.emergency_vehicle else 0.0
        ]
        
        return features
    
    def train_ml_model(self):
        """
        Train the machine learning model using historical traffic data.
        """
        if len(self.traffic_history) < 50:
            logger.warning("Insufficient data for ML training")
            return
        
        # Prepare training data
        features = []
        targets = []
        
        for i in range(len(self.traffic_history) - 1):
            current = self.traffic_history[i]
            next_data = self.traffic_history[i + 1]
            
            # Find corresponding opposite direction data
            opposite_direction = Direction.EAST_WEST if current.direction == Direction.NORTH_SOUTH else Direction.NORTH_SOUTH
            opposite_data = None
            
            # Find closest opposite direction data point
            for j in range(max(0, i-5), min(len(self.traffic_history), i+5)):
                if self.traffic_history[j].direction == opposite_direction:
                    opposite_data = self.traffic_history[j]
                    break
            
            if opposite_data:
                feature_vector = self._extract_features(current, opposite_data)
                features.append(feature_vector)
                
                # Target: optimal green time adjustment
                optimal_adjustment = self._calculate_optimal_adjustment(current, next_data)
                targets.append(optimal_adjustment)
        
        if len(features) > 10:
            # Train the model
            features_array = np.array(features)
            targets_array = np.array(targets)
            
            self.scaler.fit(features_array)
            features_scaled = self.scaler.transform(features_array)
            
            self.ml_model.fit(features_scaled, targets_array)
            self.model_trained = True
            
            logger.info(f"ML model trained with {len(features)} samples")
        else:
            logger.warning("Insufficient valid data for ML training")
    
    def _calculate_optimal_adjustment(self, current_data: TrafficData, next_data: TrafficData) -> float:
        """
        Calculate optimal timing adjustment based on traffic flow changes.
        
        Args:
            current_data: Current traffic data
            next_data: Next traffic data point
            
        Returns:
            Optimal adjustment value
        """
        # Simple heuristic: adjust based on traffic change
        traffic_change = next_data.vehicle_count - current_data.vehicle_count
        density_change = next_data.density - current_data.density
        
        # Positive adjustment for increasing traffic, negative for decreasing
        adjustment = (traffic_change * 0.5) + (density_change * 10)
        
        # Clamp adjustment to reasonable range
        return max(-10, min(10, adjustment))
    
    def handle_emergency_vehicle(self, direction: Direction):
        """
        Handle emergency vehicle detection by prioritizing the direction.
        
        Args:
            direction: Direction where emergency vehicle is detected
        """
        logger.info(f"Emergency vehicle detected in {direction.value}")
        self.emergency_mode = True
        
        # Immediately switch to green for emergency direction
        self.current_state[direction] = LightState.GREEN
        opposite_direction = Direction.EAST_WEST if direction == Direction.NORTH_SOUTH else Direction.NORTH_SOUTH
        self.current_state[opposite_direction] = LightState.RED
        
        # Set emergency mode timer
        threading.Timer(30.0, self._clear_emergency_mode).start()
    
    def _clear_emergency_mode(self):
        """Clear emergency mode after timeout."""
        self.emergency_mode = False
        logger.info("Emergency mode cleared")
    
    def get_current_state(self) -> Dict[Direction, LightState]:
        """
        Get current traffic light states.
        
        Returns:
            Dictionary mapping directions to light states
        """
        return self.current_state.copy()
    
    def get_performance_metrics(self) -> Dict[str, float]:
        """
        Get performance metrics for the traffic control system.
        
        Returns:
            Dictionary containing performance metrics
        """
        if self.cycle_count > 0:
            avg_wait_time = self.total_wait_time / self.cycle_count
        else:
            avg_wait_time = 0.0
        
        # Calculate efficiency score based on traffic flow
        recent_traffic = self.traffic_history[-10:] if len(self.traffic_history) >= 10 else self.traffic_history
        if recent_traffic:
            avg_density = sum(data.density for data in recent_traffic) / len(recent_traffic)
            self.efficiency_score = max(0, 100 - (avg_density * 10))  # Higher density = lower efficiency
        
        return {
            'cycle_count': self.cycle_count,
            'average_wait_time': avg_wait_time,
            'efficiency_score': self.efficiency_score,
            'emergency_activations': sum(1 for data in self.traffic_history if data.emergency_vehicle),
            'ml_model_trained': self.model_trained
        }
    
    def save_model(self, filepath: str):
        """
        Save the trained ML model to file.
        
        Args:
            filepath: Path to save the model
        """
        if self.model_trained:
            model_data = {
                'model': self.ml_model,
                'scaler': self.scaler,
                'intersection_id': self.intersection_id
            }
            with open(filepath, 'wb') as f:
                pickle.dump(model_data, f)
            logger.info(f"Model saved to {filepath}")
    
    def load_model(self, filepath: str):
        """
        Load a trained ML model from file.
        
        Args:
            filepath: Path to load the model from
        """
        if os.path.exists(filepath):
            with open(filepath, 'rb') as f:
                model_data = pickle.load(f)
            self.ml_model = model_data['model']
            self.scaler = model_data['scaler']
            self.model_trained = True
            logger.info(f"Model loaded from {filepath}")
        else:
            logger.warning(f"Model file not found: {filepath}")

def simulate_traffic_control_cycle():
    """
    Simulate a complete traffic control cycle with adaptive timing.
    """
    controller = AdaptiveTrafficController("DEMO_INTERSECTION")
    
    # Simulate traffic data for different scenarios
    scenarios = [
        # Morning rush hour - heavy north-south traffic
        (Direction.NORTH_SOUTH, 25, 0.8, False),
        (Direction.EAST_WEST, 10, 0.3, False),
        
        # Afternoon - balanced traffic
        (Direction.NORTH_SOUTH, 15, 0.5, False),
        (Direction.EAST_WEST, 18, 0.6, False),
        
        # Emergency vehicle scenario
        (Direction.NORTH_SOUTH, 20, 0.7, True),
        (Direction.EAST_WEST, 12, 0.4, False),
        
        # Evening rush hour - heavy east-west traffic
        (Direction.NORTH_SOUTH, 8, 0.2, False),
        (Direction.EAST_WEST, 30, 0.9, False),
    ]
    
    print("AI-Based Adaptive Traffic Light Control System")
    print("=" * 50)
    
    for i, (direction, vehicle_count, density, emergency) in enumerate(scenarios):
        print(f"\nScenario {i+1}: {direction.value}")
        print(f"Vehicle Count: {vehicle_count}, Density: {density:.1f}, Emergency: {emergency}")
        
        # Update traffic data
        controller.update_traffic_data(direction, vehicle_count, density, emergency)
        
        # Calculate adaptive timing
        timing = controller.calculate_adaptive_timing(direction)
        
        print(f"Calculated Green Time: {timing.green_duration} seconds")
        print(f"Yellow Time: {timing.yellow_duration} seconds")
        
        # Simulate some history for ML training
        time.sleep(0.1)  # Small delay to create different timestamps
    
    # Train ML model
    controller.train_ml_model()
    
    # Display performance metrics
    metrics = controller.get_performance_metrics()
    print(f"\nPerformance Metrics:")
    for metric, value in metrics.items():
        print(f"{metric.replace('_', ' ').title()}: {value}")
    
    return controller

if __name__ == "__main__":
    simulate_traffic_control_cycle()


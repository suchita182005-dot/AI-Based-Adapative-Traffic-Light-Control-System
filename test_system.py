"""
Test Suite for AI-Based Adaptive Traffic Light Control System
This module contains comprehensive test cases for all system components.
"""

import unittest
import sys
import os
import time
import numpy as np
from unittest.mock import Mock, patch

# Add src directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from vehicle_detection import VehicleDetector, simulate_traffic_data
from traffic_controller import AdaptiveTrafficController, Direction, LightState, TrafficData
from main_system import TrafficLightSystem

class TestVehicleDetection(unittest.TestCase):
    """Test cases for vehicle detection module."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.detector = VehicleDetector()
    
    def test_detector_initialization(self):
        """Test vehicle detector initialization."""
        self.assertIsNotNone(self.detector.bg_subtractor)
        self.assertEqual(self.detector.min_contour_area, 500)
        self.assertEqual(self.detector.max_contour_area, 50000)
        self.assertEqual(self.detector.vehicle_count, 0)
    
    def test_traffic_data_simulation(self):
        """Test traffic data simulation."""
        traffic_data = simulate_traffic_data()
        
        # Check that all expected scenarios are present
        expected_scenarios = [
            'north_south_morning', 'north_south_afternoon', 'north_south_evening', 'north_south_night',
            'east_west_morning', 'east_west_afternoon', 'east_west_evening', 'east_west_night'
        ]
        
        for scenario in expected_scenarios:
            self.assertIn(scenario, traffic_data)
            self.assertEqual(len(traffic_data[scenario]), 10)
            self.assertTrue(all(isinstance(count, int) for count in traffic_data[scenario]))
    
    def test_density_calculation(self):
        """Test traffic density calculation."""
        # Create a mock frame
        mock_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        
        # Test density calculation
        density = self.detector.get_traffic_density(mock_frame)
        self.assertIsInstance(density, float)
        self.assertGreaterEqual(density, 0.0)

class TestTrafficController(unittest.TestCase):
    """Test cases for traffic controller module."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.controller = AdaptiveTrafficController("TEST_INTERSECTION")
    
    def test_controller_initialization(self):
        """Test traffic controller initialization."""
        self.assertEqual(self.controller.intersection_id, "TEST_INTERSECTION")
        self.assertEqual(self.controller.min_green_time, 10)
        self.assertEqual(self.controller.max_green_time, 60)
        self.assertEqual(self.controller.default_green_time, 30)
        self.assertFalse(self.controller.model_trained)
    
    def test_traffic_data_update(self):
        """Test traffic data update functionality."""
        # Update traffic data
        self.controller.update_traffic_data(Direction.NORTH_SOUTH, 15, 0.5, False)
        
        # Check that data was updated
        ns_data = self.controller.current_traffic_data[Direction.NORTH_SOUTH]
        self.assertEqual(ns_data.vehicle_count, 15)
        self.assertEqual(ns_data.density, 0.5)
        self.assertFalse(ns_data.emergency_vehicle)
        self.assertEqual(ns_data.direction, Direction.NORTH_SOUTH)
    
    def test_adaptive_timing_calculation(self):
        """Test adaptive timing calculation."""
        # Set up traffic data
        self.controller.update_traffic_data(Direction.NORTH_SOUTH, 20, 0.6, False)
        self.controller.update_traffic_data(Direction.EAST_WEST, 10, 0.3, False)
        
        # Calculate timing
        timing = self.controller.calculate_adaptive_timing(Direction.NORTH_SOUTH)
        
        # Check timing constraints
        self.assertGreaterEqual(timing.green_duration, self.controller.min_green_time)
        self.assertLessEqual(timing.green_duration, self.controller.max_green_time)
        self.assertEqual(timing.yellow_duration, 3)
    
    def test_emergency_vehicle_handling(self):
        """Test emergency vehicle detection and handling."""
        # Trigger emergency vehicle
        self.controller.update_traffic_data(Direction.NORTH_SOUTH, 15, 0.5, True)
        
        # Check emergency mode
        self.assertTrue(self.controller.emergency_mode)
        
        # Check that green time is extended for emergency
        timing = self.controller.calculate_adaptive_timing(Direction.NORTH_SOUTH)
        self.assertGreaterEqual(timing.green_duration, 45)
    
    def test_ml_model_training(self):
        """Test machine learning model training."""
        # Generate sufficient training data
        for i in range(60):
            vehicle_count = np.random.randint(5, 30)
            density = vehicle_count / 100.0
            direction = Direction.NORTH_SOUTH if i % 2 == 0 else Direction.EAST_WEST
            self.controller.update_traffic_data(direction, vehicle_count, density, False)
            time.sleep(0.01)  # Small delay for different timestamps
        
        # Train model
        self.controller.train_ml_model()
        
        # Check that model was trained
        self.assertTrue(self.controller.model_trained)
    
    def test_performance_metrics(self):
        """Test performance metrics calculation."""
        # Add some traffic data
        self.controller.update_traffic_data(Direction.NORTH_SOUTH, 15, 0.5, False)
        self.controller.update_traffic_data(Direction.EAST_WEST, 20, 0.6, True)
        
        # Get metrics
        metrics = self.controller.get_performance_metrics()
        
        # Check metrics structure
        expected_keys = ['cycle_count', 'average_wait_time', 'efficiency_score', 
                        'emergency_activations', 'ml_model_trained']
        for key in expected_keys:
            self.assertIn(key, metrics)
        
        # Check emergency activation count
        self.assertEqual(metrics['emergency_activations'], 1)

class TestMainSystem(unittest.TestCase):
    """Test cases for main system integration."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.system = TrafficLightSystem("TEST_SYSTEM")
    
    def test_system_initialization(self):
        """Test system initialization."""
        self.assertEqual(self.system.intersection_id, "TEST_SYSTEM")
        self.assertIsNotNone(self.system.vehicle_detector)
        self.assertIsNotNone(self.system.traffic_controller)
        self.assertFalse(self.system.running)
        self.assertTrue(self.system.simulation_mode)
    
    def test_system_status(self):
        """Test system status reporting."""
        status = self.system.get_system_status()
        
        # Check status structure
        expected_keys = ['intersection_id', 'running', 'simulation_mode', 
                        'current_light_state', 'performance_metrics', 'total_cycles_logged']
        for key in expected_keys:
            self.assertIn(key, status)
        
        # Check initial values
        self.assertEqual(status['intersection_id'], "TEST_SYSTEM")
        self.assertFalse(status['running'])
        self.assertTrue(status['simulation_mode'])

class TestSystemIntegration(unittest.TestCase):
    """Integration tests for the complete system."""
    
    def test_end_to_end_simulation(self):
        """Test complete system simulation."""
        system = TrafficLightSystem("INTEGRATION_TEST")
        
        # Mock the simulation to run faster
        original_sleep = time.sleep
        time.sleep = lambda x: None  # Mock sleep to speed up test
        
        try:
            # Run a short simulation
            system.simulation_mode = True
            system.running = True
            
            # Simulate a few cycles
            for period in ['morning', 'afternoon']:
                traffic_data = simulate_traffic_data()
                ns_data = traffic_data.get(f'north_south_{period}', [15] * 3)[:3]
                ew_data = traffic_data.get(f'east_west_{period}', [15] * 3)[:3]
                
                for i in range(len(ns_data)):
                    ns_count = ns_data[i]
                    ew_count = ew_data[i]
                    ns_density = ns_count / 100.0
                    ew_density = ew_count / 100.0
                    
                    system.traffic_controller.update_traffic_data(
                        Direction.NORTH_SOUTH, ns_count, ns_density, False
                    )
                    system.traffic_controller.update_traffic_data(
                        Direction.EAST_WEST, ew_count, ew_density, False
                    )
                    
                    ns_timing = system.traffic_controller.calculate_adaptive_timing(Direction.NORTH_SOUTH)
                    ew_timing = system.traffic_controller.calculate_adaptive_timing(Direction.EAST_WEST)
                    
                    system._log_system_state(period, ns_count, ew_count, ns_timing, ew_timing)
            
            # Check that data was logged
            self.assertGreater(len(system.system_log), 0)
            
            # Check system status
            status = system.get_system_status()
            self.assertGreater(status['total_cycles_logged'], 0)
            
        finally:
            # Restore original sleep function
            time.sleep = original_sleep
            system.stop_system()

class TestPerformanceValidation(unittest.TestCase):
    """Test cases for performance validation."""
    
    def test_timing_constraints(self):
        """Test that timing constraints are always respected."""
        controller = AdaptiveTrafficController("PERFORMANCE_TEST")
        
        # Test various traffic scenarios
        test_scenarios = [
            (5, 0.1),    # Low traffic
            (15, 0.5),   # Medium traffic
            (30, 0.8),   # High traffic
            (50, 1.0),   # Very high traffic
        ]
        
        for vehicle_count, density in test_scenarios:
            controller.update_traffic_data(Direction.NORTH_SOUTH, vehicle_count, density, False)
            controller.update_traffic_data(Direction.EAST_WEST, vehicle_count // 2, density / 2, False)
            
            timing = controller.calculate_adaptive_timing(Direction.NORTH_SOUTH)
            
            # Validate constraints
            self.assertGreaterEqual(timing.green_duration, controller.min_green_time,
                                  f"Green time {timing.green_duration} below minimum for scenario {vehicle_count}, {density}")
            self.assertLessEqual(timing.green_duration, controller.max_green_time,
                               f"Green time {timing.green_duration} above maximum for scenario {vehicle_count}, {density}")
    
    def test_emergency_response_time(self):
        """Test emergency vehicle response time."""
        controller = AdaptiveTrafficController("EMERGENCY_TEST")
        
        # Normal traffic
        controller.update_traffic_data(Direction.NORTH_SOUTH, 15, 0.5, False)
        normal_timing = controller.calculate_adaptive_timing(Direction.NORTH_SOUTH)
        
        # Emergency traffic
        controller.update_traffic_data(Direction.NORTH_SOUTH, 15, 0.5, True)
        emergency_timing = controller.calculate_adaptive_timing(Direction.NORTH_SOUTH)
        
        # Emergency should get longer green time
        self.assertGreaterEqual(emergency_timing.green_duration, 45)
        self.assertTrue(controller.emergency_mode)
    
    def test_system_efficiency(self):
        """Test overall system efficiency."""
        system = TrafficLightSystem("EFFICIENCY_TEST")
        
        # Simulate traffic and measure efficiency
        total_vehicles = 0
        total_green_time = 0
        
        for _ in range(10):
            ns_count = np.random.randint(10, 25)
            ew_count = np.random.randint(10, 25)
            
            system.traffic_controller.update_traffic_data(Direction.NORTH_SOUTH, ns_count, ns_count/100.0, False)
            system.traffic_controller.update_traffic_data(Direction.EAST_WEST, ew_count, ew_count/100.0, False)
            
            ns_timing = system.traffic_controller.calculate_adaptive_timing(Direction.NORTH_SOUTH)
            ew_timing = system.traffic_controller.calculate_adaptive_timing(Direction.EAST_WEST)
            
            total_vehicles += ns_count + ew_count
            total_green_time += ns_timing.green_duration + ew_timing.green_duration
        
        # Calculate efficiency (vehicles per second of green time)
        efficiency = total_vehicles / total_green_time if total_green_time > 0 else 0
        
        # Efficiency should be reasonable (at least 0.3 vehicles per second)
        self.assertGreater(efficiency, 0.3)

def run_performance_tests():
    """Run performance tests and generate results."""
    print("Running AI-Based Adaptive Traffic Light Control System Tests")
    print("=" * 60)
    
    # Create test suite
    test_suite = unittest.TestSuite()
    
    # Add test cases
    test_classes = [
        TestVehicleDetection,
        TestTrafficController,
        TestMainSystem,
        TestSystemIntegration,
        TestPerformanceValidation
    ]
    
    for test_class in test_classes:
        tests = unittest.TestLoader().loadTestsFromTestCase(test_class)
        test_suite.addTests(tests)
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(test_suite)
    
    # Generate test report
    test_report = {
        'total_tests': result.testsRun,
        'failures': len(result.failures),
        'errors': len(result.errors),
        'success_rate': ((result.testsRun - len(result.failures) - len(result.errors)) / result.testsRun * 100) if result.testsRun > 0 else 0,
        'test_details': {
            'vehicle_detection': 'PASSED' if not any('TestVehicleDetection' in str(failure[0]) for failure in result.failures + result.errors) else 'FAILED',
            'traffic_controller': 'PASSED' if not any('TestTrafficController' in str(failure[0]) for failure in result.failures + result.errors) else 'FAILED',
            'main_system': 'PASSED' if not any('TestMainSystem' in str(failure[0]) for failure in result.failures + result.errors) else 'FAILED',
            'system_integration': 'PASSED' if not any('TestSystemIntegration' in str(failure[0]) for failure in result.failures + result.errors) else 'FAILED',
            'performance_validation': 'PASSED' if not any('TestPerformanceValidation' in str(failure[0]) for failure in result.failures + result.errors) else 'FAILED'
        }
    }
    
    return test_report, result.wasSuccessful()

if __name__ == "__main__":
    report, success = run_performance_tests()
    
    print(f"\nTest Summary:")
    print(f"Total Tests: {report['total_tests']}")
    print(f"Failures: {report['failures']}")
    print(f"Errors: {report['errors']}")
    print(f"Success Rate: {report['success_rate']:.1f}%")
    
    print(f"\nModule Test Results:")
    for module, status in report['test_details'].items():
        print(f"{module.replace('_', ' ').title()}: {status}")
    
    if success:
        print("\nAll tests passed successfully!")
    else:
        print("\nSome tests failed. Please check the output above for details.")


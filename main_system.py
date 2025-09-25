"""
Main System Integration Module for AI-Based Adaptive Traffic Light Control System
This module integrates all components and provides the main system interface.
"""

import time
import threading
import logging
import json
from datetime import datetime
from typing import Dict, List, Optional
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from dataclasses import asdict

from vehicle_detection import VehicleDetector, simulate_traffic_data
from traffic_controller import AdaptiveTrafficController, Direction, LightState, TrafficData

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class TrafficLightSystem:
    """
    Main system class that integrates vehicle detection and traffic control.
    """
    
    def __init__(self, intersection_id: str = "MAIN_INTERSECTION"):
        """
        Initialize the complete traffic light system.
        
        Args:
            intersection_id: Unique identifier for the intersection
        """
        self.intersection_id = intersection_id
        self.vehicle_detector = VehicleDetector()
        self.traffic_controller = AdaptiveTrafficController(intersection_id)
        
        # System state
        self.running = False
        self.simulation_mode = True  # Use simulated data for demonstration
        
        # Data collection
        self.system_log: List[Dict] = []
        self.performance_data: List[Dict] = []
        
        # Timing
        self.cycle_duration = 10  # seconds per detection cycle
        self.last_cycle_time = time.time()
        
    def start_system(self):
        """Start the traffic light system."""
        self.running = True
        logger.info(f"Starting AI-Based Adaptive Traffic Light Control System for {self.intersection_id}")
        
        if self.simulation_mode:
            self._run_simulation()
        else:
            self._run_real_time()
    
    def stop_system(self):
        """Stop the traffic light system."""
        self.running = False
        logger.info("Traffic light system stopped")
    
    def _run_simulation(self):
        """Run the system in simulation mode using generated traffic data."""
        logger.info("Running in simulation mode")
        
        # Get simulated traffic data
        traffic_data = simulate_traffic_data()
        
        # Simulate different time periods
        time_periods = ['morning', 'afternoon', 'evening', 'night']
        
        for period in time_periods:
            logger.info(f"Simulating {period} traffic conditions")
            
            # Get traffic data for this period
            ns_data = traffic_data.get(f'north_south_{period}', [15] * 10)
            ew_data = traffic_data.get(f'east_west_{period}', [15] * 10)
            
            for i in range(min(len(ns_data), len(ew_data))):
                if not self.running:
                    break
                
                # Simulate vehicle detection
                ns_count = ns_data[i]
                ew_count = ew_data[i]
                
                # Calculate density (vehicles per unit area)
                ns_density = ns_count / 100.0  # Normalize to 0-1 range
                ew_density = ew_count / 100.0
                
                # Randomly add emergency vehicles (5% chance)
                ns_emergency = np.random.random() < 0.05
                ew_emergency = np.random.random() < 0.05
                
                # Update traffic controller
                self.traffic_controller.update_traffic_data(
                    Direction.NORTH_SOUTH, ns_count, ns_density, ns_emergency
                )
                self.traffic_controller.update_traffic_data(
                    Direction.EAST_WEST, ew_count, ew_density, ew_emergency
                )
                
                # Calculate adaptive timing
                ns_timing = self.traffic_controller.calculate_adaptive_timing(Direction.NORTH_SOUTH)
                ew_timing = self.traffic_controller.calculate_adaptive_timing(Direction.EAST_WEST)
                
                # Log system state
                self._log_system_state(period, ns_count, ew_count, ns_timing, ew_timing)
                
                # Simulate cycle delay
                time.sleep(0.5)  # Faster simulation
        
        # Train ML model with collected data
        self.traffic_controller.train_ml_model()
        
        # Generate performance report
        self._generate_performance_report()
    
    def _run_real_time(self):
        """Run the system in real-time mode (would use actual cameras/sensors)."""
        logger.info("Running in real-time mode")
        
        while self.running:
            current_time = time.time()
            
            if current_time - self.last_cycle_time >= self.cycle_duration:
                # In real implementation, this would process camera feeds
                # For now, we'll use placeholder values
                
                # Placeholder vehicle detection (would be replaced with actual detection)
                ns_count = np.random.randint(5, 30)
                ew_count = np.random.randint(5, 30)
                ns_density = ns_count / 100.0
                ew_density = ew_count / 100.0
                
                # Update traffic controller
                self.traffic_controller.update_traffic_data(
                    Direction.NORTH_SOUTH, ns_count, ns_density
                )
                self.traffic_controller.update_traffic_data(
                    Direction.EAST_WEST, ew_count, ew_density
                )
                
                self.last_cycle_time = current_time
            
            time.sleep(1)  # Check every second
    
    def _log_system_state(self, period: str, ns_count: int, ew_count: int, ns_timing, ew_timing):
        """Log the current system state."""
        log_entry = {
            'timestamp': datetime.now().isoformat(),
            'period': period,
            'north_south': {
                'vehicle_count': ns_count,
                'green_time': ns_timing.green_duration,
                'density': ns_count / 100.0
            },
            'east_west': {
                'vehicle_count': ew_count,
                'green_time': ew_timing.green_duration,
                'density': ew_count / 100.0
            },
            'total_cycle_time': ns_timing.green_duration + ew_timing.green_duration + 6  # +6 for yellow/red
        }
        
        self.system_log.append(log_entry)
        
        # Log to console
        logger.info(f"Period: {period} | NS: {ns_count} vehicles ({ns_timing.green_duration}s green) | "
                   f"EW: {ew_count} vehicles ({ew_timing.green_duration}s green)")
    
    def _generate_performance_report(self):
        """Generate a comprehensive performance report."""
        logger.info("Generating performance report...")
        
        if not self.system_log:
            logger.warning("No data available for performance report")
            return
        
        # Calculate performance metrics
        metrics = self.traffic_controller.get_performance_metrics()
        
        # Analyze traffic patterns
        periods = {}
        for entry in self.system_log:
            period = entry['period']
            if period not in periods:
                periods[period] = {'ns_counts': [], 'ew_counts': [], 'ns_green_times': [], 'ew_green_times': []}
            
            periods[period]['ns_counts'].append(entry['north_south']['vehicle_count'])
            periods[period]['ew_counts'].append(entry['east_west']['vehicle_count'])
            periods[period]['ns_green_times'].append(entry['north_south']['green_time'])
            periods[period]['ew_green_times'].append(entry['east_west']['green_time'])
        
        # Create performance summary
        performance_summary = {
            'intersection_id': self.intersection_id,
            'total_cycles': len(self.system_log),
            'ml_model_trained': metrics['ml_model_trained'],
            'efficiency_score': metrics['efficiency_score'],
            'emergency_activations': metrics['emergency_activations'],
            'period_analysis': {}
        }
        
        for period, data in periods.items():
            performance_summary['period_analysis'][period] = {
                'avg_ns_vehicles': np.mean(data['ns_counts']),
                'avg_ew_vehicles': np.mean(data['ew_counts']),
                'avg_ns_green_time': np.mean(data['ns_green_times']),
                'avg_ew_green_time': np.mean(data['ew_green_times']),
                'total_cycles': len(data['ns_counts'])
            }
        
        # Save performance data
        with open('/home/ubuntu/traffic_light_system/results/performance_summary.json', 'w') as f:
            json.dump(performance_summary, f, indent=2)
        
        # Generate visualizations
        self._create_visualizations()
        
        logger.info("Performance report generated successfully")
        return performance_summary
    
    def _create_visualizations(self):
        """Create visualization charts for the system performance."""
        if not self.system_log:
            return
        
        # Set up the plotting style
        plt.style.use('seaborn-v0_8')
        sns.set_palette("husl")
        
        # Extract data for plotting
        periods = []
        ns_vehicles = []
        ew_vehicles = []
        ns_green_times = []
        ew_green_times = []
        
        for entry in self.system_log:
            periods.append(entry['period'])
            ns_vehicles.append(entry['north_south']['vehicle_count'])
            ew_vehicles.append(entry['east_west']['vehicle_count'])
            ns_green_times.append(entry['north_south']['green_time'])
            ew_green_times.append(entry['east_west']['green_time'])
        
        # Create subplots
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('AI-Based Adaptive Traffic Light Control System Performance', fontsize=16, fontweight='bold')
        
        # Plot 1: Vehicle counts by period
        period_data = {}
        for i, period in enumerate(periods):
            if period not in period_data:
                period_data[period] = {'ns': [], 'ew': []}
            period_data[period]['ns'].append(ns_vehicles[i])
            period_data[period]['ew'].append(ew_vehicles[i])
        
        period_names = list(period_data.keys())
        ns_means = [np.mean(period_data[p]['ns']) for p in period_names]
        ew_means = [np.mean(period_data[p]['ew']) for p in period_names]
        
        x = np.arange(len(period_names))
        width = 0.35
        
        ax1.bar(x - width/2, ns_means, width, label='North-South', alpha=0.8)
        ax1.bar(x + width/2, ew_means, width, label='East-West', alpha=0.8)
        ax1.set_xlabel('Time Period')
        ax1.set_ylabel('Average Vehicle Count')
        ax1.set_title('Average Vehicle Count by Time Period')
        ax1.set_xticks(x)
        ax1.set_xticklabels(period_names)
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Green light duration by period
        ns_green_means = [np.mean([entry['north_south']['green_time'] for entry in self.system_log if entry['period'] == p]) for p in period_names]
        ew_green_means = [np.mean([entry['east_west']['green_time'] for entry in self.system_log if entry['period'] == p]) for p in period_names]
        
        ax2.bar(x - width/2, ns_green_means, width, label='North-South', alpha=0.8)
        ax2.bar(x + width/2, ew_green_means, width, label='East-West', alpha=0.8)
        ax2.set_xlabel('Time Period')
        ax2.set_ylabel('Average Green Time (seconds)')
        ax2.set_title('Average Green Light Duration by Time Period')
        ax2.set_xticks(x)
        ax2.set_xticklabels(period_names)
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # Plot 3: Vehicle count vs Green time correlation
        ax3.scatter(ns_vehicles, ns_green_times, alpha=0.6, label='North-South', s=50)
        ax3.scatter(ew_vehicles, ew_green_times, alpha=0.6, label='East-West', s=50)
        ax3.set_xlabel('Vehicle Count')
        ax3.set_ylabel('Green Time (seconds)')
        ax3.set_title('Vehicle Count vs Green Light Duration')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # Add trend line
        all_vehicles = ns_vehicles + ew_vehicles
        all_green_times = ns_green_times + ew_green_times
        z = np.polyfit(all_vehicles, all_green_times, 1)
        p = np.poly1d(z)
        ax3.plot(sorted(all_vehicles), p(sorted(all_vehicles)), "r--", alpha=0.8, linewidth=2)
        
        # Plot 4: System efficiency over time
        cycle_numbers = list(range(1, len(self.system_log) + 1))
        total_vehicles = [entry['north_south']['vehicle_count'] + entry['east_west']['vehicle_count'] for entry in self.system_log]
        total_green_time = [entry['north_south']['green_time'] + entry['east_west']['green_time'] for entry in self.system_log]
        efficiency = [vehicles / green_time if green_time > 0 else 0 for vehicles, green_time in zip(total_vehicles, total_green_time)]
        
        ax4.plot(cycle_numbers, efficiency, marker='o', linewidth=2, markersize=4, alpha=0.7)
        ax4.set_xlabel('Cycle Number')
        ax4.set_ylabel('Efficiency (Vehicles/Green Time)')
        ax4.set_title('System Efficiency Over Time')
        ax4.grid(True, alpha=0.3)
        
        # Add moving average
        if len(efficiency) >= 5:
            moving_avg = np.convolve(efficiency, np.ones(5)/5, mode='valid')
            ax4.plot(cycle_numbers[2:-2], moving_avg, 'r-', linewidth=3, alpha=0.8, label='Moving Average')
            ax4.legend()
        
        plt.tight_layout()
        plt.savefig('/home/ubuntu/traffic_light_system/results/performance_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # Create a second figure for traffic flow patterns
        fig2, (ax5, ax6) = plt.subplots(1, 2, figsize=(15, 6))
        fig2.suptitle('Traffic Flow Patterns Analysis', fontsize=16, fontweight='bold')
        
        # Plot 5: Traffic flow heatmap
        period_order = ['morning', 'afternoon', 'evening', 'night']
        heatmap_data = []
        
        for period in period_order:
            period_entries = [entry for entry in self.system_log if entry['period'] == period]
            if period_entries:
                avg_ns = np.mean([entry['north_south']['vehicle_count'] for entry in period_entries])
                avg_ew = np.mean([entry['east_west']['vehicle_count'] for entry in period_entries])
                heatmap_data.append([avg_ns, avg_ew])
            else:
                heatmap_data.append([0, 0])
        
        heatmap_data = np.array(heatmap_data)
        im = ax5.imshow(heatmap_data, cmap='YlOrRd', aspect='auto')
        ax5.set_xticks([0, 1])
        ax5.set_xticklabels(['North-South', 'East-West'])
        ax5.set_yticks(range(len(period_order)))
        ax5.set_yticklabels(period_order)
        ax5.set_title('Traffic Density Heatmap')
        
        # Add text annotations
        for i in range(len(period_order)):
            for j in range(2):
                text = ax5.text(j, i, f'{heatmap_data[i, j]:.1f}', 
                               ha="center", va="center", color="black", fontweight='bold')
        
        plt.colorbar(im, ax=ax5, label='Average Vehicle Count')
        
        # Plot 6: Adaptive timing effectiveness
        fixed_timing = 30  # Assume fixed 30-second green time
        adaptive_times = ns_green_times + ew_green_times
        time_savings = [fixed_timing - adaptive_time for adaptive_time in adaptive_times]
        
        ax6.hist(time_savings, bins=20, alpha=0.7, edgecolor='black')
        ax6.set_xlabel('Time Savings (seconds)')
        ax6.set_ylabel('Frequency')
        ax6.set_title('Adaptive Timing vs Fixed Timing')
        ax6.axvline(np.mean(time_savings), color='red', linestyle='--', linewidth=2, 
                   label=f'Average Savings: {np.mean(time_savings):.1f}s')
        ax6.legend()
        ax6.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('/home/ubuntu/traffic_light_system/results/traffic_flow_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info("Visualizations created successfully")
    
    def get_system_status(self) -> Dict:
        """Get current system status."""
        current_state = self.traffic_controller.get_current_state()
        metrics = self.traffic_controller.get_performance_metrics()
        
        return {
            'intersection_id': self.intersection_id,
            'running': self.running,
            'simulation_mode': self.simulation_mode,
            'current_light_state': {
                'north_south': current_state[Direction.NORTH_SOUTH].value,
                'east_west': current_state[Direction.EAST_WEST].value
            },
            'performance_metrics': metrics,
            'total_cycles_logged': len(self.system_log)
        }

def main():
    """Main function to run the traffic light system."""
    print("AI-Based Adaptive Traffic Light Control System")
    print("=" * 60)
    print("Initializing system...")
    
    # Create and start the system
    system = TrafficLightSystem("DEMO_INTERSECTION_001")
    
    try:
        # Run the simulation
        system.start_system()
        
        # Display final status
        status = system.get_system_status()
        print("\nSystem Status:")
        print(f"Intersection ID: {status['intersection_id']}")
        print(f"Total Cycles: {status['total_cycles_logged']}")
        print(f"ML Model Trained: {status['performance_metrics']['ml_model_trained']}")
        print(f"Efficiency Score: {status['performance_metrics']['efficiency_score']:.1f}")
        
        print("\nSystem demonstration completed successfully!")
        print("Check the results folder for performance analysis and visualizations.")
        
    except KeyboardInterrupt:
        print("\nSystem interrupted by user")
    finally:
        system.stop_system()

if __name__ == "__main__":
    main()


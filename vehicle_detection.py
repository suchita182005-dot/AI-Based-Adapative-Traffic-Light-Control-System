"""
Vehicle Detection Module for AI-Based Adaptive Traffic Light Control System
This module implements computer vision algorithms for detecting and counting vehicles
at traffic intersections using OpenCV.
"""

import cv2
import numpy as np
import time
from typing import List, Tuple, Dict
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class VehicleDetector:
    """
    A class for detecting and counting vehicles in video streams or images.
    Uses background subtraction and contour detection for vehicle identification.
    """
    
    def __init__(self, min_contour_area: int = 500, max_contour_area: int = 50000):
        """
        Initialize the vehicle detector.
        
        Args:
            min_contour_area: Minimum area for a contour to be considered a vehicle
            max_contour_area: Maximum area for a contour to be considered a vehicle
        """
        self.min_contour_area = min_contour_area
        self.max_contour_area = max_contour_area
        
        # Initialize background subtractor
        self.bg_subtractor = cv2.createBackgroundSubtractorMOG2(
            detectShadows=True, varThreshold=50, history=500
        )
        
        # Morphological operations kernel
        self.kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        
        # Vehicle count tracking
        self.vehicle_count = 0
        self.detection_line_y = None
        self.tracked_vehicles = []
        
    def preprocess_frame(self, frame: np.ndarray) -> np.ndarray:
        """
        Preprocess the input frame for vehicle detection.
        
        Args:
            frame: Input video frame
            
        Returns:
            Preprocessed frame
        """
        # Apply Gaussian blur to reduce noise
        blurred = cv2.GaussianBlur(frame, (5, 5), 0)
        
        # Apply background subtraction
        fg_mask = self.bg_subtractor.apply(blurred)
        
        # Remove shadows
        fg_mask[fg_mask == 127] = 0
        
        # Morphological operations to clean up the mask
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, self.kernel)
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_CLOSE, self.kernel)
        
        return fg_mask
    
    def detect_vehicles(self, frame: np.ndarray) -> List[Tuple[int, int, int, int]]:
        """
        Detect vehicles in the given frame.
        
        Args:
            frame: Input video frame
            
        Returns:
            List of bounding boxes (x, y, w, h) for detected vehicles
        """
        # Preprocess the frame
        fg_mask = self.preprocess_frame(frame)
        
        # Find contours
        contours, _ = cv2.findContours(fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        vehicles = []
        for contour in contours:
            area = cv2.contourArea(contour)
            
            # Filter contours based on area
            if self.min_contour_area < area < self.max_contour_area:
                x, y, w, h = cv2.boundingRect(contour)
                
                # Additional filtering based on aspect ratio
                aspect_ratio = w / h
                if 0.5 < aspect_ratio < 3.0:  # Reasonable aspect ratio for vehicles
                    vehicles.append((x, y, w, h))
        
        return vehicles
    
    def count_vehicles_crossing_line(self, frame: np.ndarray, detection_line_y: int) -> int:
        """
        Count vehicles crossing a detection line.
        
        Args:
            frame: Input video frame
            detection_line_y: Y-coordinate of the detection line
            
        Returns:
            Number of vehicles that crossed the line
        """
        self.detection_line_y = detection_line_y
        vehicles = self.detect_vehicles(frame)
        
        current_centroids = []
        for x, y, w, h in vehicles:
            centroid_x = x + w // 2
            centroid_y = y + h // 2
            current_centroids.append((centroid_x, centroid_y))
        
        # Simple tracking: count vehicles crossing the line
        crossing_count = 0
        for centroid in current_centroids:
            if abs(centroid[1] - detection_line_y) < 10:  # Within 10 pixels of the line
                crossing_count += 1
        
        return crossing_count
    
    def get_traffic_density(self, frame: np.ndarray, roi: Tuple[int, int, int, int] = None) -> float:
        """
        Calculate traffic density in the given frame.
        
        Args:
            frame: Input video frame
            roi: Region of interest (x, y, w, h). If None, uses entire frame
            
        Returns:
            Traffic density (vehicles per unit area)
        """
        if roi:
            x, y, w, h = roi
            frame_roi = frame[y:y+h, x:x+w]
            area = w * h
        else:
            frame_roi = frame
            area = frame.shape[0] * frame.shape[1]
        
        vehicles = self.detect_vehicles(frame_roi)
        vehicle_count = len(vehicles)
        
        # Calculate density (vehicles per 1000 square pixels)
        density = (vehicle_count * 1000) / area
        
        return density
    
    def draw_detections(self, frame: np.ndarray, vehicles: List[Tuple[int, int, int, int]]) -> np.ndarray:
        """
        Draw bounding boxes around detected vehicles.
        
        Args:
            frame: Input video frame
            vehicles: List of vehicle bounding boxes
            
        Returns:
            Frame with drawn detections
        """
        result_frame = frame.copy()
        
        for x, y, w, h in vehicles:
            cv2.rectangle(result_frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(result_frame, 'Vehicle', (x, y - 10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        
        # Draw detection line if set
        if self.detection_line_y:
            cv2.line(result_frame, (0, self.detection_line_y), 
                    (frame.shape[1], self.detection_line_y), (255, 0, 0), 2)
        
        # Display vehicle count
        cv2.putText(result_frame, f'Vehicles: {len(vehicles)}', (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        
        return result_frame

def simulate_traffic_data() -> Dict[str, List[int]]:
    """
    Simulate traffic data for different time periods and directions.
    
    Returns:
        Dictionary containing simulated traffic counts for different scenarios
    """
    # Simulate traffic data for a 4-way intersection
    traffic_data = {
        'north_south_morning': [15, 18, 22, 25, 30, 28, 24, 20, 16, 12],
        'north_south_afternoon': [12, 14, 16, 20, 25, 30, 35, 32, 28, 22],
        'north_south_evening': [25, 30, 35, 40, 45, 42, 38, 32, 26, 20],
        'north_south_night': [5, 4, 3, 2, 1, 2, 3, 4, 5, 6],
        
        'east_west_morning': [20, 25, 30, 35, 40, 38, 32, 28, 22, 18],
        'east_west_afternoon': [18, 20, 22, 26, 30, 28, 24, 20, 16, 14],
        'east_west_evening': [30, 35, 40, 45, 50, 48, 42, 36, 30, 24],
        'east_west_night': [8, 6, 4, 3, 2, 3, 4, 6, 8, 10]
    }
    
    return traffic_data

if __name__ == "__main__":
    # Example usage
    detector = VehicleDetector()
    
    # Simulate some traffic data
    traffic_data = simulate_traffic_data()
    
    print("AI-Based Adaptive Traffic Light Control System")
    print("Vehicle Detection Module Initialized")
    print("\nSimulated Traffic Data:")
    
    for scenario, counts in traffic_data.items():
        avg_count = sum(counts) / len(counts)
        print(f"{scenario.replace('_', ' ').title()}: Average {avg_count:.1f} vehicles")


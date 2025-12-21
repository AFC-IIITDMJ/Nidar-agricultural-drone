import math
import queue
import threading
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
from dronekit import Command, LocationGlobalRelative, Vehicle, VehicleMode, connect
from pymavlink import mavutil

# ============================================================================
# CONFIGURATION
# ============================================================================


@dataclass
class CameraConfig:
    """Camera configuration parameters."""

    width: int = 640
    height: int = 480
    hfov: float = 1.047
    vfov: float = 1.047 * (480.0 / 640.0)
    focal_length_x: float = 915.3
    focal_length_y: float = 914.0
    center_x: float = 320.0
    center_y: float = 240.0


@dataclass
class DetectionConfig:
    """Detection algorithm configuration."""

    brown_lower: np.ndarray = field(default_factory=lambda: np.array([10, 50, 20]))
    brown_upper: np.ndarray = field(default_factory=lambda: np.array([25, 255, 200]))
    min_blob_area: int = 150
    confidence_threshold: float = 0.75
    deduplication_threshold: float = 2.5
    min_detection_frames: int = 2
    motion_threshold: float = 3.0


@dataclass
class SurveyConfig:
    """Survey mission configuration."""

    altitude: float = 10.0
    speed: float = 2.0
    image_overlap: float = 0.75
    connection_string: str = "udp:127.0.0.1:14550"
    survey_area: List[LocationGlobalRelative] = field(default_factory=list)


@dataclass
class VisualizationConfig:
    """Visualization settings."""

    show_live_feed: bool = True
    display_scale: float = 1.0


# ============================================================================
# DATA MODELS
# ============================================================================


@dataclass
class Detection:
    """Represents a single crop detection."""

    cx: int
    cy: int
    confidence: float
    area: float
    bbox: Tuple[int, int, int, int]
    contour: Optional[np.ndarray] = None


@dataclass
class CropLocation:
    """Represents a confirmed crop location with GPS coordinates."""

    lat: float
    lon: float
    confidence: float
    pixel_x: int
    pixel_y: int
    area: float
    altitude: float
    heading: float
    velocity: float = 0.0
    detection_count: int = 1
    last_seen_altitude: Optional[float] = None
    last_seen_heading: Optional[float] = None


@dataclass
class DroneState:
    """Current drone telemetry state."""

    location: LocationGlobalRelative
    heading: float
    altitude: float
    velocity: float


@dataclass
class Statistics:
    """Mission statistics."""

    total_frames: int = 0
    processed_frames: int = 0
    total_detections: int = 0
    stabilization_stops: int = 0


# ============================================================================
# UTILITIES
# ============================================================================


class GeoUtils:
    """Geographic calculation utilities."""

    EARTH_RADIUS = 6378137.0

    @staticmethod
    def get_distance_metres(
        loc1: LocationGlobalRelative, loc2: LocationGlobalRelative
    ) -> float:
        """Calculate distance between two GPS coordinates in meters."""
        dlat = loc2.lat - loc1.lat
        dlong = loc2.lon - loc1.lon
        return math.sqrt((dlat * dlat) + (dlong * dlong)) * 1.113195e5

    @staticmethod
    def get_bearing(
        loc1: LocationGlobalRelative, loc2: LocationGlobalRelative
    ) -> float:
        """Calculate bearing between two GPS coordinates."""
        off_x = loc2.lon - loc1.lon
        off_y = loc2.lat - loc1.lat
        bearing = 90.00 + math.atan2(-off_y, off_x) * 57.2957795
        if bearing < 0:
            bearing += 360.00
        return bearing

    @staticmethod
    def get_location_metres(
        original: LocationGlobalRelative, d_north: float, d_east: float
    ) -> LocationGlobalRelative:
        """Get GPS location offset by meters."""
        d_lat = d_north / GeoUtils.EARTH_RADIUS
        d_lon = d_east / (
            GeoUtils.EARTH_RADIUS * math.cos(math.pi * original.lat / 180)
        )

        new_lat = original.lat + (d_lat * 180 / math.pi)
        new_lon = original.lon + (d_lon * 180 / math.pi)
        return LocationGlobalRelative(new_lat, new_lon, original.alt)


class CameraModel:
    """Camera projection and coordinate transformation."""

    def __init__(self, config: CameraConfig):
        self.config = config

    def calculate_ground_coverage(self, altitude: float) -> Tuple[float, float]:
        """Calculate ground coverage area at given altitude."""
        ground_width = 2 * altitude * math.tan(self.config.hfov / 2)
        ground_height = 2 * altitude * math.tan(self.config.vfov / 2)
        return ground_width, ground_height

    def pixel_to_gps(
        self, drone_state: DroneState, pixel_x: int, pixel_y: int
    ) -> LocationGlobalRelative:
        """Convert pixel coordinates to GPS location with motion compensation."""
        altitude = drone_state.altitude
        heading = drone_state.heading
        velocity = drone_state.velocity

        # Pixel offset from center
        x_offset_pixels = pixel_x - self.config.center_x
        y_offset_pixels = pixel_y - self.config.center_y

        # Convert to meters using focal length
        x_offset_meters = (x_offset_pixels / self.config.focal_length_x) * altitude
        y_offset_meters = (y_offset_pixels / self.config.focal_length_y) * altitude

        # Body frame: forward is X, right is Y
        body_x = -y_offset_meters
        body_y = x_offset_meters

        # Apply heading rotation to world frame
        heading_rad = math.radians(heading)
        d_north = body_x * math.cos(heading_rad) - body_y * math.sin(heading_rad)
        d_east = body_x * math.sin(heading_rad) + body_y * math.cos(heading_rad)

        # Motion compensation
        if velocity > 0.5:
            latency = 0.15
            velocity_north = velocity * math.cos(heading_rad)
            velocity_east = velocity * math.sin(heading_rad)
            d_north -= velocity_north * latency
            d_east -= velocity_east * latency

        return GeoUtils.get_location_metres(drone_state.location, d_north, d_east)


# ============================================================================
# DETECTION ENGINE
# ============================================================================


class DetectionAlgorithm(ABC):
    """Abstract base class for detection algorithms."""

    @abstractmethod
    def detect(self, image: np.ndarray) -> Tuple[List[Detection], np.ndarray]:
        """Detect crops in image and return detections with debug mask."""
        pass


class BrownCropDetector(DetectionAlgorithm):
    """HSV-based brown crop detection algorithm."""

    def __init__(self, config: DetectionConfig):
        self.config = config

    def detect(self, image: np.ndarray) -> Tuple[List[Detection], np.ndarray]:
        """Detect brown crops using HSV color segmentation."""
        detections = []

        # Convert to HSV
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

        # Create mask
        mask = cv2.inRange(hsv, self.config.brown_lower, self.config.brown_upper)

        # Morphological operations
        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
        mask = cv2.dilate(mask, kernel, iterations=2)

        # Find contours
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Sort contours left-to-right
        if len(contours) > 0:
            contours = sorted(contours, key=lambda c: cv2.boundingRect(c)[0])

        for contour in contours:
            area = cv2.contourArea(contour)

            if area > self.config.min_blob_area:
                M = cv2.moments(contour)
                if M["m00"] != 0:
                    cx = int(M["m10"] / M["m00"])
                    cy = int(M["m01"] / M["m00"])

                    x, y, w, h = cv2.boundingRect(contour)
                    aspect_ratio = float(w) / h

                    if 0.3 < aspect_ratio < 3.0:
                        confidence = min(area / 1000.0, 1.0)

                        if confidence >= self.config.confidence_threshold:
                            detections.append(
                                Detection(
                                    cx=cx,
                                    cy=cy,
                                    confidence=confidence,
                                    area=area,
                                    bbox=(x, y, w, h),
                                    contour=contour,
                                )
                            )

        return detections, mask


class DetectionTracker:
    """Track detections across multiple frames for stability."""

    def __init__(self, min_frames: int = 2):
        self.min_frames = min_frames
        self.pending: Dict[str, Dict] = {}

    def update(self, detections: List[Detection], frame_key: int) -> List[Detection]:
        """Update tracking and return stable detections."""
        stable_detections = []

        # Update pending detections
        for det in detections:
            det_id = f"{det.cx // 20}_{det.cy // 20}"

            if det_id not in self.pending:
                self.pending[det_id] = {
                    "frames": [det],
                    "first_seen": frame_key,
                    "last_seen": frame_key,
                }
            else:
                self.pending[det_id]["frames"].append(det)
                self.pending[det_id]["last_seen"] = frame_key

        # Check for stable detections
        for det_id, track in list(self.pending.items()):
            frame_count = len(track["frames"])

            if frame_count >= self.min_frames:
                # Average detection across frames
                avg_cx = sum(d.cx for d in track["frames"]) / frame_count
                avg_cy = sum(d.cy for d in track["frames"]) / frame_count
                avg_conf = sum(d.confidence for d in track["frames"]) / frame_count

                latest = track["frames"][-1]
                stable_detections.append(
                    Detection(
                        cx=int(avg_cx),
                        cy=int(avg_cy),
                        confidence=avg_conf,
                        area=latest.area,
                        bbox=latest.bbox,
                        contour=latest.contour,
                    )
                )

                del self.pending[det_id]

            # Clean up old pending
            elif track["last_seen"] < frame_key - 10:
                del self.pending[det_id]

        return stable_detections


class CropLocationManager:
    """Manage detected crop locations with deduplication."""

    def __init__(self, threshold_meters: float = 2.5):
        self.threshold = threshold_meters
        self.crops: List[CropLocation] = []

    def add_or_update(self, crop: CropLocation) -> bool:
        """Add new crop or update existing. Returns True if new."""
        is_dup, idx = self._find_duplicate(crop)

        if is_dup:
            self.crops[idx] = self._merge_duplicate(self.crops[idx], crop)
            return False
        else:
            self.crops.append(crop)
            return True

    def _find_duplicate(self, crop: CropLocation) -> Tuple[bool, int]:
        """Check if crop is duplicate of existing detection."""
        new_loc = LocationGlobalRelative(crop.lat, crop.lon, 0)

        for idx, existing in enumerate(self.crops):
            existing_loc = LocationGlobalRelative(existing.lat, existing.lon, 0)
            distance = GeoUtils.get_distance_metres(existing_loc, new_loc)

            if distance <= self.threshold:
                return True, idx

        return False, -1

    def _merge_duplicate(
        self, existing: CropLocation, new: CropLocation
    ) -> CropLocation:
        """Merge duplicate detection with weighted average."""
        existing.detection_count += 1

        total_weight = existing.confidence + new.confidence
        existing.lat = (
            existing.lat * existing.confidence + new.lat * new.confidence
        ) / total_weight
        existing.lon = (
            existing.lon * existing.confidence + new.lon * new.confidence
        ) / total_weight

        existing.confidence = max(existing.confidence, new.confidence)
        existing.last_seen_altitude = new.altitude
        existing.last_seen_heading = new.heading

        return existing

    def get_all(self) -> List[CropLocation]:
        """Get all detected crops."""
        return self.crops.copy()


# ============================================================================
# VISUALIZATION
# ============================================================================


class Visualizer:
    """Handle visualization and display of detections."""

    def __init__(self, config: VisualizationConfig, camera_config: CameraConfig):
        self.config = config
        self.camera_config = camera_config
        self.display_queue = queue.Queue(maxsize=2)
        self.active = threading.Event()
        self.thread: Optional[threading.Thread] = None

    def start(self):
        """Start display thread."""
        if self.config.show_live_feed:
            self.active.set()
            self.thread = threading.Thread(target=self._display_loop, daemon=True)
            self.thread.start()

    def stop(self):
        """Stop display thread."""
        self.active.clear()
        if self.thread:
            self.thread.join(timeout=2)

    def update(self, annotated: np.ndarray, mask: np.ndarray):
        """Update display with new frame."""
        if self.active.is_set() and not self.display_queue.full():
            self.display_queue.put({"annotated": annotated, "mask": mask})

    def draw_detections(
        self,
        image: np.ndarray,
        detections: List[Detection],
        drone_state: DroneState,
        stats: Statistics,
    ) -> np.ndarray:
        """Draw detections and HUD on image."""
        annotated = image.copy()

        # Draw detections
        for det in detections:
            color = (0, 255, 0) if det.confidence > 0.8 else (0, 255, 255)
            x, y, w, h = det.bbox

            cv2.rectangle(annotated, (x, y), (x + w, y + h), color, 2)
            cv2.circle(annotated, (det.cx, det.cy), 5, color, -1)

            text = f"{det.confidence:.2f}"
            cv2.putText(
                annotated, text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2
            )

        # Draw crosshair
        center_x = self.camera_config.width // 2
        center_y = self.camera_config.height // 2
        cv2.line(
            annotated,
            (center_x - 20, center_y),
            (center_x + 20, center_y),
            (255, 255, 255),
            1,
        )
        cv2.line(
            annotated,
            (center_x, center_y - 20),
            (center_x, center_y + 20),
            (255, 255, 255),
            1,
        )

        # Draw HUD
        hud_data = [
            f"Alt: {drone_state.altitude:.1f}m",
            f"Hdg: {drone_state.heading:.0f}deg",
            f"Vel: {drone_state.velocity:.1f}m/s",
            f"Lat: {drone_state.location.lat:.6f}",
            f"Lon: {drone_state.location.lon:.6f}",
            f"Detections: {len(detections)}",
            f"Unique: {stats.total_detections}",
        ]

        y_pos = 20
        for text in hud_data:
            cv2.putText(
                annotated,
                text,
                (10, y_pos),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 0, 0),
                1,
            )
            y_pos += 20

        return annotated

    def _display_loop(self):
        """Display thread main loop."""
        cv2.namedWindow("Survey Feed", cv2.WINDOW_NORMAL)
        cv2.namedWindow("Detection Mask", cv2.WINDOW_NORMAL)

        width = int(self.camera_config.width * self.config.display_scale)
        height = int(self.camera_config.height * self.config.display_scale)
        cv2.resizeWindow("Survey Feed", width, height)
        cv2.resizeWindow("Detection Mask", width, height)

        print("Display thread started - Press 'q' to stop")

        while self.active.is_set():
            try:
                data = self.display_queue.get(timeout=0.1)
                cv2.imshow("Survey Feed", data["annotated"])
                cv2.imshow("Detection Mask", data["mask"])

                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    self.active.clear()
                    break
                elif key == ord("s"):
                    timestamp = int(time.time())
                    cv2.imwrite(f"screenshot_{timestamp}.jpg", data["annotated"])
                    cv2.imwrite(f"mask_{timestamp}.jpg", data["mask"])
                    print(f"Screenshots saved: {timestamp}")

            except queue.Empty:
                continue
            except Exception as e:
                print(f"Display error: {e}")

        cv2.destroyAllWindows()


# ============================================================================
# IMAGE PROCESSING PIPELINE
# ============================================================================


class ImageProcessor:
    """Main image processing pipeline."""

    def __init__(
        self,
        vehicle: Vehicle,
        camera_config: CameraConfig,
        detection_config: DetectionConfig,
        vis_config: VisualizationConfig,
    ):
        self.vehicle = vehicle
        self.camera_config = camera_config
        self.detection_config = detection_config

        self.detector = BrownCropDetector(detection_config)
        self.tracker = DetectionTracker(detection_config.min_detection_frames)
        self.crop_manager = CropLocationManager(
            detection_config.deduplication_threshold
        )
        self.camera_model = CameraModel(camera_config)
        self.visualizer = Visualizer(vis_config, camera_config)

        self.image_queue = queue.Queue(maxsize=10)
        self.stats = Statistics()
        self.active = threading.Event()
        self.frame_counter = 0

        self.ros_thread: Optional[threading.Thread] = None
        self.processing_thread: Optional[threading.Thread] = None

    def start(self):
        """Start image processing pipeline."""
        self.active.set()

        # Start ROS2 image subscriber
        self.ros_thread = threading.Thread(
            target=self._ros_subscriber_loop, daemon=True
        )
        self.ros_thread.start()
        time.sleep(3)

        # Start processing thread
        self.processing_thread = threading.Thread(
            target=self._processing_loop, daemon=True
        )
        self.processing_thread.start()

        # Start visualizer
        self.visualizer.start()

    def stop(self):
        """Stop image processing pipeline."""
        self.active.clear()
        self.visualizer.stop()

        if self.processing_thread:
            self.processing_thread.join(timeout=5)

    def get_detected_crops(self) -> List[CropLocation]:
        """Get all detected crop locations."""
        return self.crop_manager.get_all()

    def _ros_subscriber_loop(self):
        """ROS2 image subscriber thread."""
        try:
            import rclpy
            from cv_bridge import CvBridge
            from rclpy.node import Node
            from sensor_msgs.msg import Image

            rclpy.init()

            class ImageSubscriber(Node):
                def __init__(self, processor):
                    super().__init__("crop_detector")
                    self.processor = processor
                    self.bridge = CvBridge()
                    self.subscription = self.create_subscription(
                        Image, "/camera", self.callback, 10
                    )
                    self.get_logger().info("✓ Image subscriber initialized")

                def callback(self, msg):
                    self.processor.stats.total_frames += 1

                    if self.processor.active.is_set():
                        try:
                            image = self.bridge.imgmsg_to_cv2(msg, "bgr8")

                            drone_state = DroneState(
                                location=self.processor.vehicle.location.global_relative_frame,
                                heading=self.processor.vehicle.heading or 0,
                                altitude=self.processor.vehicle.location.global_relative_frame.alt,
                                velocity=self.processor.vehicle.groundspeed or 0,
                            )

                            if not self.processor.image_queue.full():
                                self.processor.image_queue.put((image, drone_state))
                        except Exception as e:
                            self.get_logger().error(f"Error: {e}")

            subscriber = ImageSubscriber(self)
            print("✓ Image processing thread started - ROS 2 connected")
            rclpy.spin(subscriber)

            subscriber.destroy_node()
            rclpy.shutdown()

        except ImportError as e:
            print(f"\n✗ ROS 2 not found: {e}")
        except Exception as e:
            print(f"✗ ROS 2 error: {e}")

    def _processing_loop(self):
        """Main image processing loop."""
        while self.active.is_set() or not self.image_queue.empty():
            try:
                image, drone_state = self.image_queue.get(timeout=1.0)
                self.stats.processed_frames += 1
                self.frame_counter += 1

                # Skip if moving too fast
                if drone_state.velocity > self.detection_config.motion_threshold:
                    continue

                # Detect crops
                detections, mask = self.detector.detect(image)

                # Track across frames
                stable_detections = self.tracker.update(detections, self.frame_counter)

                # Process stable detections
                for det in stable_detections:
                    gps_loc = self.camera_model.pixel_to_gps(
                        drone_state, det.cx, det.cy
                    )

                    crop = CropLocation(
                        lat=gps_loc.lat,
                        lon=gps_loc.lon,
                        confidence=det.confidence,
                        pixel_x=det.cx,
                        pixel_y=det.cy,
                        area=det.area,
                        altitude=drone_state.altitude,
                        heading=drone_state.heading,
                        velocity=drone_state.velocity,
                    )

                    if self.crop_manager.add_or_update(crop):
                        self.stats.total_detections += 1
                        print(
                            f"✓ NEW crop #{len(self.crop_manager.crops)}: "
                            f"{crop.lat:.7f}, {crop.lon:.7f} "
                            f"(conf: {crop.confidence:.2f})"
                        )

                # Visualize
                viz_detections = detections if detections else stable_detections
                annotated = self.visualizer.draw_detections(
                    image, viz_detections, drone_state, self.stats
                )
                self.visualizer.update(
                    annotated, cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
                )

            except queue.Empty:
                continue
            except Exception as e:
                print(f"Processing error: {e}")
                import traceback

                traceback.print_exc()


# ============================================================================
# MISSION CONTROL
# ============================================================================


class MissionPlanner:
    """Generate survey mission waypoints."""

    def __init__(self, camera_config: CameraConfig):
        self.camera_config = camera_config
        self.camera_model = CameraModel(camera_config)

    def generate_survey_pattern(
        self, survey_area: List[LocationGlobalRelative], altitude: float, overlap: float
    ) -> List[LocationGlobalRelative]:
        """Generate lawnmower survey pattern."""
        waypoints = []

        ground_width, _ = self.camera_model.calculate_ground_coverage(altitude)
        spacing = ground_width * (1 - overlap)

        min_lat = min(p.lat for p in survey_area)
        max_lat = max(p.lat for p in survey_area)
        min_lon = min(p.lon for p in survey_area)
        max_lon = max(p.lon for p in survey_area)

        area_height_m = GeoUtils.get_distance_metres(
            LocationGlobalRelative(min_lat, min_lon, 0),
            LocationGlobalRelative(max_lat, min_lon, 0),
        )
        area_width_m = GeoUtils.get_distance_metres(
            LocationGlobalRelative(min_lat, min_lon, 0),
            LocationGlobalRelative(min_lat, max_lon, 0),
        )

        num_passes = int(area_width_m / spacing) + 1

        print(f"\nSurvey Pattern:")
        print(f"  Area: {area_width_m:.1f}m x {area_height_m:.1f}m")
        print(f"  Passes: {num_passes}")
        print(f"  Spacing: {spacing:.1f}m")

        start_location = LocationGlobalRelative(min_lat, min_lon, altitude)

        for i in range(num_passes):
            offset_east = i * spacing
            if i % 2 == 0:
                wp_start = GeoUtils.get_location_metres(start_location, 0, offset_east)
                wp_end = GeoUtils.get_location_metres(wp_start, area_height_m, 0)
            else:
                wp_start = GeoUtils.get_location_metres(
                    start_location, area_height_m, offset_east
                )
                wp_end = GeoUtils.get_location_metres(wp_start, -area_height_m, 0)

            waypoints.append(wp_start)
            waypoints.append(wp_end)

        return waypoints


# ============================================================================
# MISSION CONTROL (CONTINUED)
# ============================================================================


class DroneController:
    """Control drone operations."""

    def __init__(self, vehicle: Vehicle):
        self.vehicle = vehicle

    def arm_and_takeoff(self, target_altitude: float) -> bool:
        """Arm vehicle and takeoff to target altitude."""
        print("\n" + "=" * 70)
        print("PHASE 1: ARM AND TAKEOFF")
        print("=" * 70)

        # Pre-arm checks
        print("\n[1/4] Pre-arm checks...")
        timeout = 60
        elapsed = 0
        while not self.vehicle.is_armable and elapsed < timeout:
            print(f"  Waiting... ({timeout - elapsed}s)", end="\r")
            time.sleep(1)
            elapsed += 1

        if not self.vehicle.is_armable:
            print("\n✗ Vehicle not armable")
            return False
        print("\n✓ Vehicle is armable")

        # Set GUIDED mode
        print("\n[2/4] Setting GUIDED mode...")
        self.vehicle.mode = VehicleMode("GUIDED")
        time.sleep(2)

        if self.vehicle.mode.name != "GUIDED":
            print("\n✗ Failed to set GUIDED mode")
            return False
        print("✓ Mode set to GUIDED")

        # Arm
        print("\n[3/4] Arming vehicle...")
        self.vehicle.armed = True
        timeout = 30
        elapsed = 0
        while not self.vehicle.armed and elapsed < timeout:
            print(f"  Waiting... ({timeout - elapsed}s)", end="\r")
            time.sleep(1)
            elapsed += 1

        if not self.vehicle.armed:
            print("\n✗ Failed to arm")
            return False
        print("\n✓ Vehicle ARMED")

        # Takeoff
        print(f"\n[4/4] Taking off to {target_altitude}m...")
        self.vehicle.simple_takeoff(target_altitude)

        while True:
            current_alt = self.vehicle.location.global_relative_frame.alt or 0
            print(f"  Altitude: {current_alt:.1f}m / {target_altitude}m", end="\r")
            if current_alt >= target_altitude * 0.95:
                break
            time.sleep(1)

        print(f"\n✓ Reached altitude: {current_alt:.1f}m")
        return True

    def upload_mission(self, waypoints: List[LocationGlobalRelative]) -> bool:
        """Upload mission waypoints to vehicle."""
        print("\n" + "=" * 70)
        print("PHASE 2: UPLOAD MISSION")
        print("=" * 70)

        cmds = self.vehicle.commands
        cmds.clear()
        cmds.download()
        cmds.wait_ready()

        for wp in waypoints:
            cmds.add(
                Command(
                    0,
                    0,
                    0,
                    mavutil.mavlink.MAV_FRAME_GLOBAL_RELATIVE_ALT,
                    mavutil.mavlink.MAV_CMD_NAV_WAYPOINT,
                    0,
                    0,
                    0,
                    2.0,
                    0,
                    float("nan"),
                    wp.lat,
                    wp.lon,
                    wp.alt,
                )
            )

        # Add RTL
        cmds.add(
            Command(
                0,
                0,
                0,
                mavutil.mavlink.MAV_FRAME_GLOBAL_RELATIVE_ALT,
                mavutil.mavlink.MAV_CMD_NAV_RETURN_TO_LAUNCH,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
            )
        )

        cmds.upload()
        cmds.wait_ready()

        print(f"✓ Mission uploaded: {len(waypoints)} waypoints")
        return True

    def execute_mission(self, speed: float) -> float:
        """Execute AUTO mission and monitor progress."""
        print("\n" + "=" * 70)
        print("PHASE 3: MISSION EXECUTION")
        print("=" * 70)

        self.vehicle.groundspeed = speed
        self.vehicle.mode = VehicleMode("AUTO")
        time.sleep(2)

        total_waypoints = len(self.vehicle.commands)
        last_waypoint = -1
        start_time = time.time()

        while self.vehicle.armed:
            current_wp = self.vehicle.commands.next

            if current_wp != last_waypoint:
                print(f"\n→ WP {current_wp + 1}/{total_waypoints}")
                last_waypoint = current_wp

            if current_wp >= total_waypoints:
                print("\n✓ All waypoints reached!")
                break

            time.sleep(1)

        elapsed = time.time() - start_time
        print(f"\n✓ Mission complete in {elapsed:.1f}s")
        return elapsed


class ResultsExporter:
    """Export detection results to file."""

    @staticmethod
    def save_to_csv(crops: List[CropLocation], filename: Optional[str] = None) -> str:
        """Save detected crops to CSV file."""
        if filename is None:
            filename = f"detected_crops_{int(time.time())}.csv"

        with open(filename, "w") as f:
            f.write(
                "ID,Latitude,Longitude,Confidence,Detection_Count,"
                "Pixel_X,Pixel_Y,Area,Heading,Altitude,Velocity\n"
            )

            for i, crop in enumerate(crops, 1):
                f.write(
                    f"{i},{crop.lat},{crop.lon},{crop.confidence},"
                    f"{crop.detection_count},{crop.pixel_x},{crop.pixel_y},"
                    f"{crop.area},{crop.heading},{crop.altitude},{crop.velocity}\n"
                )

        return filename


# ============================================================================
# SURVEY ORCHESTRATOR
# ============================================================================


class CropSurveySystem:
    """Main orchestrator for crop survey missions."""

    def __init__(
        self,
        survey_config: SurveyConfig,
        camera_config: CameraConfig,
        detection_config: DetectionConfig,
        vis_config: VisualizationConfig,
    ):
        self.survey_config = survey_config
        self.camera_config = camera_config
        self.detection_config = detection_config
        self.vis_config = vis_config

        self.vehicle: Optional[Vehicle] = None
        self.controller: Optional[DroneController] = None
        self.planner: Optional[MissionPlanner] = None
        self.processor: Optional[ImageProcessor] = None

    def connect(self) -> bool:
        """Connect to vehicle."""
        print("\n" + "=" * 70)
        print("CROP DETECTION SURVEY SYSTEM")
        print("=" * 70)
        print(f"\nConnecting to: {self.survey_config.connection_string}")

        try:
            self.vehicle = connect(
                self.survey_config.connection_string, wait_ready=True, timeout=60
            )
            print("✓ Connected to vehicle")

            self.controller = DroneController(self.vehicle)
            self.planner = MissionPlanner(self.camera_config)
            self.processor = ImageProcessor(
                self.vehicle, self.camera_config, self.detection_config, self.vis_config
            )

            return True

        except Exception as e:
            print(f"✗ Connection failed: {e}")
            return False

    def execute_survey(self) -> bool:
        """Execute complete survey mission."""
        if not self.vehicle or not self.controller:
            print("✗ Not connected to vehicle")
            return False

        try:
            # Arm and takeoff
            if not self.controller.arm_and_takeoff(self.survey_config.altitude):
                return False

            time.sleep(3)

            # Generate and upload mission
            waypoints = self.planner.generate_survey_pattern(
                self.survey_config.survey_area,
                self.survey_config.altitude,
                self.survey_config.image_overlap,
            )

            if not self.controller.upload_mission(waypoints):
                return False

            # Start image processing
            self.processor.start()

            # Execute mission
            elapsed = self.controller.execute_mission(self.survey_config.speed)

            # Stop processing
            time.sleep(2)
            self.processor.stop()

            # Report results
            self._report_results(elapsed)

            return True

        except KeyboardInterrupt:
            print("\n\n⚠ Mission interrupted by user")
            if self.processor:
                self.processor.stop()
            return False

        except Exception as e:
            print(f"\n✗ Error during survey: {e}")
            import traceback

            traceback.print_exc()
            return False

    def _report_results(self, elapsed_time: float):
        """Print and save final results."""
        crops = self.processor.get_detected_crops()
        stats = self.processor.stats

        print("\n" + "=" * 70)
        print("FINAL RESULTS")
        print("=" * 70)
        print(f"\nMission Statistics:")
        print(f"  Duration: {elapsed_time:.1f}s")
        print(f"  Total frames: {stats.total_frames}")
        print(f"  Processed frames: {stats.processed_frames}")
        print(f"  Unique crops detected: {len(crops)}")

        print(f"\nDetected Crops:")
        for i, crop in enumerate(crops, 1):
            print(
                f"  {i}. Lat: {crop.lat:.7f}, Lon: {crop.lon:.7f}, "
                f"Conf: {crop.confidence:.2f}, Seen: {crop.detection_count}x"
            )

        # Save to CSV
        filename = ResultsExporter.save_to_csv(crops)
        print(f"\n✓ Results saved to: {filename}")

    def disconnect(self):
        """Disconnect from vehicle and cleanup."""
        if self.processor:
            self.processor.stop()

        if self.vehicle:
            print("\nClosing vehicle connection...")
            self.vehicle.close()
            print("✓ Done")


# ============================================================================
# MAIN ENTRY POINT
# ============================================================================


def main():
    """Main entry point for the application."""

    # Define survey area
    survey_area = [
        LocationGlobalRelative(-35.363262, 149.165402, 0),
        LocationGlobalRelative(-35.363262, 149.165612, 0),
        LocationGlobalRelative(-35.363091, 149.165612, 0),
        LocationGlobalRelative(-35.363091, 149.165402, 0),
    ]

    # Create configurations
    survey_config = SurveyConfig(
        altitude=10.0,
        speed=2.0,
        image_overlap=0.75,
        connection_string="udp:127.0.0.1:14550",
        survey_area=survey_area,
    )

    camera_config = CameraConfig()
    detection_config = DetectionConfig()
    vis_config = VisualizationConfig(show_live_feed=True)

    # Create and run survey system
    system = CropSurveySystem(
        survey_config, camera_config, detection_config, vis_config
    )

    try:
        if system.connect():
            system.execute_survey()
    finally:
        system.disconnect()


if __name__ == "__main__":
    main()
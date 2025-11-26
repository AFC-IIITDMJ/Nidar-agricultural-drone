
import os
os.environ['QT_QPA_PLATFORM'] = 'xcb'

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import threading
import time
import numpy as np
import cv2
from dronekit import connect, VehicleMode, LocationGlobalRelative, Command
from pymavlink import mavutil
import math
import json
import sys
from collections import deque
from sklearn.cluster import DBSCAN
import queue
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class Config:
    SCOUT_DRONE = "udp:127.0.0.1:14550"
    TREATMENT_DRONE = "udp:127.0.0.1:14560"
    SURVEY_ALT = 2.0
    TREATMENT_ALT = 1.0 
    HOVER_TIME = 5.0
    ALIGNMENT_TIME = 3.0
    FIELD_START_X = 0.0
    FIELD_START_Y = 2.0
    FIELD_END_X = 3.0
    FIELD_END_Y = 5.0
    LINE_SPACING = 0.3
    SAFE_ZONE_X = 5.0
    SAFE_ZONE_Y = 5.0
    CAMERA_TOPIC = "/camera"
    HFOV = math.radians(60)
    VFOV = math.radians(45)
    LOWER_HSV_INFECTED = np.array([10, 80, 40])
    UPPER_HSV_INFECTED = np.array([35, 255, 255])
    MIN_CONTOUR_AREA = 300
    CENTER_THRESHOLD = 0.5
    TREATMENT_EXCLUSION_RADIUS = 2.0
    GPS_PRECISION_THRESHOLD = 0.000018
    INSTANT_CONFIRMATION = True
    DETECTION_COOLDOWN = 2.0
    ALIGNMENT_PRECISION = 1.0

stop_event = threading.Event()
scout_camera_frame = None
scout_camera_metadata = None
scout_state = None
treatment_state = None

frame_lock = threading.Lock()
state_lock = threading.Lock()
coordination_queue = queue.Queue()
treated_crops_lock = threading.Lock()

scout_mission_state = {
    'mode': 'WAYPOINT_FOLLOWING',
    'current_waypoint': 0,
    'waypoints': [],
    'return_waypoint': None
}

treatment_mission_state = {
    'mode': 'IDLE',
    'target_location': None
}

treated_crops_database = []
all_detected_crops = []
last_detection_location = None
last_detection_time = 0

detection_statistics = {
    'detections_found': 0,
    'crops_confirmed': 0,
    'crops_treated': 0,
    'crops_skipped_already_treated': 0,
    'instant_confirmations': 0
}

def add_treated_crop(lat, lon):
    global treated_crops_database
    with treated_crops_lock:
        treated_crops_database.append([lat, lon])
        detection_statistics['crops_treated'] += 1
        logger.info(f"Added to treated crops database: {lat:.8f}, {lon:.8f}")
        logger.info(f"Total treated crops: {len(treated_crops_database)}")

def is_crop_already_treated(lat, lon):
    global treated_crops_database
    with treated_crops_lock:
        for treated_lat, treated_lon in treated_crops_database:
            distance = CoordinateTransformer.get_distance_meters(
                (lat, lon), (treated_lat, treated_lon)
            )
            if distance <= Config.TREATMENT_EXCLUSION_RADIUS:
                detection_statistics['crops_skipped_already_treated'] += 1
                logger.debug(f"SKIPPING: Crop within {distance:.1f}m of treated crop")
                return True
    return False

def is_too_close_to_last_detection(lat, lon):
    global last_detection_location, last_detection_time
    current_time = time.time()
    if last_detection_location is None:
        return False
    if current_time - last_detection_time < Config.DETECTION_COOLDOWN:
        distance = CoordinateTransformer.get_distance_meters(
            (lat, lon), last_detection_location
        )
        if distance < Config.TREATMENT_EXCLUSION_RADIUS:
            logger.debug(f"COOLDOWN: Too soon since last detection ({current_time - last_detection_time:.1f}s)")
            return True
    return False

def update_last_detection(lat, lon):
    global last_detection_location, last_detection_time
    last_detection_location = (lat, lon)
    last_detection_time = time.time()

def get_treated_crops_summary():
    with treated_crops_lock:
        return {
            'total_treated': len(treated_crops_database),
            'coordinates': treated_crops_database.copy()
        }

class ScoutCameraNode(Node):
    def __init__(self):
        super().__init__('scout_camera_node')
        self.bridge = CvBridge()
        self.subscription = self.create_subscription(
            Image, 
            Config.CAMERA_TOPIC, 
            self.camera_callback, 
            10
        )
        logger.info("Scout camera node initialized with INSTANT confirmation")
    
    def camera_callback(self, msg):
        global scout_camera_frame, scout_camera_metadata, scout_state
        try:
            cv_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
            if frame_lock.acquire(blocking=False):
                try:
                    scout_camera_frame = cv_image.copy()
                    if state_lock.acquire(blocking=False):
                        try:
                            if scout_state:
                                scout_camera_metadata = {
                                    'location': scout_state['location'],
                                    'attitude': scout_state['attitude'],
                                    'timestamp': time.time()
                                }
                            else:
                                scout_camera_metadata = None
                        finally:
                            state_lock.release()
                finally:
                    frame_lock.release()
        except Exception as e:
            self.get_logger().error(f"Camera callback error: {e}")

class CoordinateTransformer:
    @staticmethod
    def get_distance_meters(coord1, coord2):
        lat1, lon1 = coord1
        lat2, lon2 = coord2
        R = 6371000
        dlat = math.radians(lat2 - lat1)
        dlon = math.radians(lon2 - lon1)
        a = (math.sin(dlat/2)**2 + 
             math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) * 
             math.sin(dlon/2)**2)
        return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))
    
    @staticmethod
    def get_precise_gps_when_centered(drone_lat, drone_lon, drone_alt):
        return drone_lat, drone_lon

class InstantConfirmationDetectionProcessor:
    def __init__(self):
        self.kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        
    def detect_infected_crops_in_center(self, frame):
        if frame is None:
            return False
        height, width = frame.shape[:2]
        center_margin = Config.CENTER_THRESHOLD
        center_x1 = int(width * (0.5 - center_margin/2))
        center_x2 = int(width * (0.5 + center_margin/2))
        center_y1 = int(height * (0.5 - center_margin/2))
        center_y2 = int(height * (0.5 + center_margin/2))
        try:
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            mask = cv2.inRange(hsv, Config.LOWER_HSV_INFECTED, Config.UPPER_HSV_INFECTED)
            center_mask = np.zeros_like(mask)
            center_mask[center_y1:center_y2, center_x1:center_x2] = mask[center_y1:center_y2, center_x1:center_x2]
            center_mask = cv2.morphologyEx(center_mask, cv2.MORPH_OPEN, self.kernel)
            center_mask = cv2.morphologyEx(center_mask, cv2.MORPH_CLOSE, self.kernel)
            contours, _ = cv2.findContours(center_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for contour in contours:
                area = cv2.contourArea(contour)
                if area >= Config.MIN_CONTOUR_AREA:
                    return True
        except Exception as e:
            logger.warning(f"Detection error: {e}")
        return False
    
    def process_frame_for_instant_confirmation(self, frame, metadata):
        if frame is None or metadata is None:
            return
        if scout_mission_state['mode'] != 'WAYPOINT_FOLLOWING':
            return
        infection_detected = self.detect_infected_crops_in_center(frame)
        if infection_detected:
            detection_statistics['detections_found'] += 1
            logger.info("INFECTION DETECTED in center region!")
            current_lat, current_lon, _ = metadata['location']
            if is_crop_already_treated(current_lat, current_lon):
                logger.debug("SKIP: Already treated")
                return
            if is_too_close_to_last_detection(current_lat, current_lon):
                logger.debug("SKIP: Too soon since last detection")
                return
            detection_statistics['crops_confirmed'] += 1
            detection_statistics['instant_confirmations'] += 1
            logger.info(f"INSTANT CONFIRMATION! New infection at {current_lat:.6f}, {current_lon:.6f}")
            logger.info(f"Confirmed: {detection_statistics['crops_confirmed']}")
            all_detected_crops.append([current_lat, current_lon])
            update_last_detection(current_lat, current_lon)
            coordination_queue.put({
                'action': 'ALIGN_OVER_INFECTION',
                'detection_location': metadata['location'],
                'timestamp': time.time(),
                'confirmation_type': 'INSTANT'
            })
            scout_mission_state['mode'] = 'ALIGNING'
            logger.info("Scout switching to ALIGNING mode for instant treatment")

class DroneOperations:
    @staticmethod
    def connect_vehicle(connection_string, timeout=60):
        logger.info(f"Connecting to vehicle at {connection_string}")
        try:
            vehicle = connect(connection_string, wait_ready=False, timeout=timeout)
            start_time = time.time()
            while time.time() - start_time < timeout:
                try:
                    if vehicle.version and vehicle.system_status.state:
                        logger.info(f"Successfully connected to {connection_string}")
                        return vehicle
                except:
                    pass
                time.sleep(0.5)
            raise RuntimeError(f"Vehicle initialization timeout for {connection_string}")
        except Exception as e:
            logger.error(f"Connection failed for {connection_string}: {e}")
            raise
    
    @staticmethod
    def arm_and_takeoff(vehicle, target_altitude, drone_name=""):
        logger.info(f"Starting {drone_name} pre-arm checks...")
        while not vehicle.is_armable:
            logger.info(f"Waiting for {drone_name} vehicle to become armable...")
            time.sleep(1)
        logger.info(f"{drone_name} Vehicle is armable")
        vehicle.mode = VehicleMode("GUIDED")
        while vehicle.mode.name != "GUIDED":
            logger.info(f"Waiting for {drone_name} GUIDED mode...")
            time.sleep(0.5)
        logger.info(f"Arming {drone_name} motors...")
        vehicle.armed = True
        while not vehicle.armed:
            logger.info(f"Waiting for {drone_name} vehicle to arm...")
            time.sleep(0.5)
        logger.info(f"{drone_name} Taking off to {target_altitude}m")
        vehicle.simple_takeoff(target_altitude)
        while True:
            current_alt = vehicle.location.global_relative_frame.alt or 0
            logger.info(f"{drone_name} Altitude: {current_alt:.1f}m")
            if current_alt >= target_altitude-1 * 0.95:
                logger.info(f"{drone_name} Target altitude reached!")
                break
            time.sleep(1)

def generate_lawnmower_waypoints(home):
    lat_factor = 1.0 / 111000.0
    lon_factor = 1.0 / (111000.0 * abs(math.cos(math.radians(home.lat))))
    waypoints = []
    north_length = Config.FIELD_END_X - Config.FIELD_START_X
    east_width = Config.FIELD_END_Y - Config.FIELD_START_Y
    num_passes = max(1, int(north_length / Config.LINE_SPACING) + 1)
    logger.info(f"Generating waypoints with instant confirmation:")
    logger.info(f"  Field: X({Config.FIELD_START_X} to {Config.FIELD_END_X}), Y({Config.FIELD_START_Y} to {Config.FIELD_END_Y})")
    logger.info(f"  Survey lines: {num_passes} passes")
    for i in range(num_passes):
        current_north = Config.FIELD_START_X + i * Config.LINE_SPACING
        row_start_lat = home.lat + current_north * lat_factor
        if i % 2 == 0:
            start_y = Config.FIELD_START_Y
            end_y = Config.FIELD_END_Y
        else:
            start_y = Config.FIELD_END_Y
            end_y = Config.FIELD_START_Y
        start_point = (row_start_lat, home.lon + start_y * lon_factor)
        end_point = (row_start_lat, home.lon + end_y * lon_factor)
        waypoints.append(start_point)
        waypoints.append(end_point)
    logger.info(f"Generated {len(waypoints)} waypoints for instant confirmation")
    return waypoints

def update_drone_state(vehicle, drone_name, stop_event):
    global scout_state, treatment_state
    while not stop_event.is_set():
        try:
            if vehicle.location.global_relative_frame:
                loc = vehicle.location.global_relative_frame
                att = vehicle.attitude
                state_info = {
                    'location': (loc.lat, loc.lon, loc.alt),
                    'attitude': (att.roll, att.pitch, att.yaw),
                    'armed': vehicle.armed,
                    'mode': str(vehicle.mode.name),
                    'timestamp': time.time()
                }
                with state_lock:
                    if drone_name == "scout":
                        scout_state = state_info
                    elif drone_name == "treatment":
                        treatment_state = state_info
        except Exception as e:
            logger.error(f"State update error for {drone_name}: {e}")
        time.sleep(0.1)

def instant_detection_thread(processor, stop_event):
    while not stop_event.is_set():
        if frame_lock.acquire(blocking=False):
            try:
                frame = scout_camera_frame.copy() if scout_camera_frame is not None else None
                metadata = scout_camera_metadata.copy() if scout_camera_metadata is not None else None
            finally:
                frame_lock.release()
            if frame is not None and metadata is not None:
                processor.process_frame_for_instant_confirmation(frame, metadata)
        time.sleep(0.1)

def scout_drone_controller(vehicle, waypoints, stop_event):
    logger.info("Scout drone controller started with INSTANT confirmation")
    scout_mission_state['waypoints'] = waypoints
    scout_mission_state['current_waypoint'] = 0
    upload_waypoint_mission(vehicle, waypoints)
    vehicle.mode = VehicleMode("AUTO")
    while vehicle.mode.name != "AUTO":
        logger.info("Waiting for AUTO mode...")
        time.sleep(0.5)
    logger.info("Scout mission started - INSTANT confirmation active")
    vehicle.commands.next = 0
    while not stop_event.is_set():
        try:
            try:
                message = coordination_queue.get(timeout=1.0)
                if message['action'] == 'ALIGN_OVER_INFECTION':
                    confirmation_type = message.get('confirmation_type', 'UNKNOWN')
                    logger.info(f"ALIGNING over {confirmation_type} infection detection")
                    vehicle.mode = VehicleMode("GUIDED")
                    while vehicle.mode.name != "GUIDED":
                        time.sleep(0.5)
                    current_loc = vehicle.location.global_relative_frame
                    alignment_target = LocationGlobalRelative(
                        current_loc.lat, current_loc.lon, Config.SURVEY_ALT
                    )
                    vehicle.simple_goto(alignment_target)
                    logger.info(f"Hovering for {Config.ALIGNMENT_TIME}s for precise GPS...")
                    alignment_start = time.time()
                    while time.time() - alignment_start < Config.ALIGNMENT_TIME:
                        vehicle.simple_goto(alignment_target)
                        time.sleep(0.5)
                    precise_loc = vehicle.location.global_relative_frame
                    precise_lat, precise_lon = CoordinateTransformer.get_precise_gps_when_centered(
                        precise_loc.lat, precise_loc.lon, precise_loc.alt
                    )
                    logger.info(f"PRECISE coordinates: {precise_lat:.8f}, {precise_lon:.8f}")
                    add_treated_crop(precise_lat, precise_lon)
                    treatment_coordination_queue.put({
                        'action': 'TREAT_INFECTION',
                        'coordinates': (precise_lat, precise_lon),
                        'confirmation_type': confirmation_type,
                        'timestamp': time.time()
                    })
                    logger.info("Treatment drone dispatched with instant coordinates")
                    scout_mission_state['mode'] = 'WAITING'
                elif message['action'] == 'CONTINUE_MISSION':
                    logger.info("Treatment arrived, resuming waypoint mission...")
                    vehicle.mode = VehicleMode("AUTO")
                    while vehicle.mode.name != "AUTO":
                        time.sleep(0.5)
                    scout_mission_state['mode'] = 'WAYPOINT_FOLLOWING'
                    logger.info("Ready for next INSTANT detection")
            except queue.Empty:
                pass
            if scout_mission_state['mode'] == 'WAYPOINT_FOLLOWING':
                current_wp = vehicle.commands.next
                total_wp = vehicle.commands.count
                if current_wp >= total_wp:
                    logger.info("Scout mission completed with instant confirmations")
                    break
                scout_mission_state['current_waypoint'] = current_wp
        except Exception as e:
            logger.error(f"Scout controller error: {e}")
            time.sleep(1)
    logger.info("Scout drone controller completed")

treatment_coordination_queue = queue.Queue()

def treatment_drone_controller(vehicle, stop_event):
    logger.info("Treatment drone ready for INSTANT response")
    home_location = vehicle.home_location or vehicle.location.global_frame
    while not stop_event.is_set():
        try:
            message = treatment_coordination_queue.get(timeout=5.0)
            if message['action'] == 'TREAT_INFECTION':
                coordinates = message['coordinates']
                lat, lon = coordinates
                confirmation_type = message.get('confirmation_type', 'UNKNOWN')
                logger.info(f"INSTANT TREATMENT: {confirmation_type} detection at {lat:.8f}, {lon:.8f}")
                if not vehicle.armed:
                    DroneOperations.arm_and_takeoff(vehicle, Config.SURVEY_ALT, "TREATMENT")
                target_survey = LocationGlobalRelative(lat, lon, Config.SURVEY_ALT)
                vehicle.simple_goto(target_survey)
                start_time = time.time()
                while time.time() - start_time < 30:
                    current_location = vehicle.location.global_relative_frame
                    if current_location:
                        distance = CoordinateTransformer.get_distance_meters(
                            (current_location.lat, current_location.lon), (lat, lon)
                        )
                        if distance < Config.ALIGNMENT_PRECISION:
                            logger.info("Treatment drone arrived at instant coordinates!")
                            break
                    time.sleep(1)
                coordination_queue.put({
                    'action': 'CONTINUE_MISSION',
                    'timestamp': time.time()
                })
                target_treatment = LocationGlobalRelative(lat, lon, Config.TREATMENT_ALT)
                vehicle.simple_goto(target_treatment)
                while True:
                    current_alt = vehicle.location.global_relative_frame.alt or 0
                    if current_alt <= Config.TREATMENT_ALT * 1.1:
                        break
                    time.sleep(0.5)
                logger.info(f"Treating {confirmation_type} infection for {Config.HOVER_TIME}s...")
                hover_start = time.time()
                while time.time() - hover_start < Config.HOVER_TIME:
                    vehicle.simple_goto(target_treatment)
                    time.sleep(0.5)
                logger.info("Instant treatment completed!")
                safe_lat = home_location.lat + (Config.SAFE_ZONE_X / 111000.0)
                safe_lon = home_location.lon + (Config.SAFE_ZONE_Y / 111000.0)
                safe_target = LocationGlobalRelative(safe_lat, safe_lon, Config.SURVEY_ALT)
                logger.info("Moving to safe area...")
                vehicle.simple_goto(safe_target)
                start_time = time.time()
                while time.time() - start_time < 20:
                    current_location = vehicle.location.global_relative_frame
                    if current_location:
                        distance = CoordinateTransformer.get_distance_meters(
                            (current_location.lat, current_location.lon), (safe_lat, safe_lon)
                        )
                        if distance < 2.0:
                            logger.info("Ready for next instant response")
                            break
                    time.sleep(1)
        except queue.Empty:
            time.sleep(1)
        except Exception as e:
            logger.error(f"Treatment controller error: {e}")
            time.sleep(1)

def upload_waypoint_mission(vehicle, waypoints):
    cmds = vehicle.commands
    cmds.clear()
    time.sleep(1)
    home_loc = vehicle.home_location or vehicle.location.global_frame
    cmds.add(Command(0, 0, 0, mavutil.mavlink.MAV_FRAME_GLOBAL_RELATIVE_ALT,
                     mavutil.mavlink.MAV_CMD_NAV_TAKEOFF, 0, 0,
                     0, 0, 0, 0, home_loc.lat, home_loc.lon, Config.SURVEY_ALT))
    for lat, lon in waypoints:
        cmds.add(Command(0, 0, 0, mavutil.mavlink.MAV_FRAME_GLOBAL_RELATIVE_ALT,
                         mavutil.mavlink.MAV_CMD_NAV_WAYPOINT, 0, 0,
                         0, 0, 0, 0, lat, lon, Config.SURVEY_ALT))
    cmds.add(Command(0, 0, 0, mavutil.mavlink.MAV_FRAME_GLOBAL,
                     mavutil.mavlink.MAV_CMD_NAV_RETURN_TO_LAUNCH, 0, 0,
                     0, 0, 0, 0, 0, 0, 0))
    cmds.upload()
    logger.info(f"Mission uploaded for instant confirmation: {cmds.count} commands")

def display_thread(processor, stop_event):
    cv2.namedWindow("Drone View", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Drone View", 600, 400)
    while not stop_event.is_set():
        if frame_lock.acquire(blocking=False):
            try:
                frame = scout_camera_frame.copy() if scout_camera_frame is not None else None
            finally:
                frame_lock.release()
        if frame is not None:
            height, width = frame.shape[:2]
            center_margin = Config.CENTER_THRESHOLD
            center_x1 = int(width * (0.5 - center_margin/2))
            center_x2 = int(width * (0.5 + center_margin/2))
            center_y1 = int(height * (0.5 - center_margin/2))
            center_y2 = int(height * (0.5 + center_margin/2))
            cv2.rectangle(frame, (center_x1, center_y1), (center_x2, center_y2), (0, 255, 0), 3)
            cv2.putText(frame, "INSTANT CONFIRMATION ZONE", (center_x1, center_y1-10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            infection_in_center = processor.detect_infected_crops_in_center(frame)
            if infection_in_center:
                cv2.rectangle(frame, (center_x1-5, center_y1-5), (center_x2+5, center_y2+5), (0, 0, 255), 3)
                cv2.putText(frame, "INSTANT CONFIRMATION!", (center_x1-50, center_y1-30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
            overlay = frame.copy()
            
            
            scout_mode = scout_mission_state['mode']
            treatment_mode = treatment_mission_state['mode']
            cv2.putText(frame, f"Scout: {scout_mode} | Treatment: {treatment_mode}", 
                       (20, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            cv2.putText(frame, f"Detections Found: {detection_statistics['detections_found']}", 
                       (20, 95), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
            cv2.putText(frame, f"INSTANT Confirmations: {detection_statistics['instant_confirmations']}", 
                       (20, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            cv2.putText(frame, f"Crops Treated: {detection_statistics['crops_treated']}", 
                       (20, 145), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
            cv2.putText(frame, f"Skipped (already treated): {detection_statistics['crops_skipped_already_treated']}", 
                       (20, 170), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 1)
            cv2.putText(frame, f"Detections: {detection_statistics['detections_found']} | Treated: {detection_statistics['crops_treated']}", (20,35), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)
            cv2.imshow("Drone View", frame)
        else:
            blank = np.zeros((600, 800, 3), np.uint8)
            cv2.putText(blank, "INSTANT CONFIRMATION", (200, 250),
                       cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
            cv2.putText(blank, "No waiting - immediate treatment trigger", (180, 330),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 1)
            cv2.imshow("Drone View", blank)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            logger.info("Display quit requested")
            stop_event.set()
            break
        time.sleep(0.03)
    cv2.destroyAllWindows()

def main():
    logger.info("="*80)
    logger.info("Dual Drone System Started")
    logger.info("="*80)
    
    
    
    
    
    
    logger.info("="*80)
    rclpy.init()
    camera_node = ScoutCameraNode()
    ros_thread = threading.Thread(target=lambda: rclpy.spin(camera_node), daemon=True)
    ros_thread.start()
    processor = InstantConfirmationDetectionProcessor()
    display_thread_handle = threading.Thread(
        target=display_thread, 
        args=(processor, stop_event), 
        daemon=True
    )
    display_thread_handle.start()
    detection_thread_handle = threading.Thread(
        target=instant_detection_thread,
        args=(processor, stop_event),
        daemon=True
    )
    detection_thread_handle.start()
    scout_vehicle = None
    treatment_vehicle = None
    try:
        logger.info("Connecting to both drones...")
        scout_vehicle = DroneOperations.connect_vehicle(Config.SCOUT_DRONE)
        treatment_vehicle = DroneOperations.connect_vehicle(Config.TREATMENT_DRONE)
        scout_state_thread = threading.Thread(
            target=update_drone_state, 
            args=(scout_vehicle, "scout", stop_event), 
            daemon=True
        )
        treatment_state_thread = threading.Thread(
            target=update_drone_state, 
            args=(treatment_vehicle, "treatment", stop_event), 
            daemon=True
        )
        scout_state_thread.start()
        treatment_state_thread.start()
        time.sleep(3)
        home_location = scout_vehicle.home_location or scout_vehicle.location.global_frame
        waypoints = generate_lawnmower_waypoints(home_location)
        logger.info("Starting scout drone with INSTANT confirmation...")
        DroneOperations.arm_and_takeoff(scout_vehicle, Config.SURVEY_ALT, "SCOUT")
        treatment_controller_thread = threading.Thread(
            target=treatment_drone_controller,
            args=(treatment_vehicle, stop_event),
            daemon=True
        )
        treatment_controller_thread.start()
        scout_drone_controller(scout_vehicle, waypoints, stop_event)
        logger.info(f"\n{'='*70}")
        logger.info("INSTANT CONFIRMATION MISSION REPORT")
        logger.info(f"{'='*70}")
        
        logger.info(f"DETECTION STATISTICS:")
        logger.info(f"    Detections found: {detection_statistics['detections_found']}")
        logger.info(f"    INSTANT confirmations: {detection_statistics['instant_confirmations']}")
        logger.info(f"    Crops treated: {detection_statistics['crops_treated']}")
        logger.info(f"    Already treated (skipped): {detection_statistics['crops_skipped_already_treated']}")
        
        
        
        treated_summary = get_treated_crops_summary()
        logger.info("\nTreated crop coordinates:")
        for i, (lat, lon) in enumerate(treated_summary['coordinates'], 1):
            logger.info(f"  {i}. Lat: {lat:.8f}, Lon: {lon:.8f}")
        logger.info(f"{'='*70}")
        results = {
            'system_version': 'instant_confirmation',
            'fix_applied': 'instant_confirmation_no_waiting',
            'confirmation_method': 'immediate_trigger_on_detection',
            'prevention_method': 'treated_crops_database_and_cooldown',
            'statistics': detection_statistics,
            'results': {
                'total_instant_confirmations': detection_statistics['instant_confirmations'],
                'total_crops_treated': detection_statistics['crops_treated'],
                'treated_coordinates': treated_summary['coordinates'],
                'timestamp': time.time()
            }
        }
        with open('instant_confirmation_results.json', 'w') as f:
            json.dump(results, f, indent=2)
        logger.info("Results saved to instant_confirmation_results.json")
    except Exception as e:
        logger.error(f"Mission error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        logger.info("Shutting down instant confirmation system...")
        stop_event.set()
        for vehicle, name in [(scout_vehicle, "Scout"), (treatment_vehicle, "Treatment")]:
            if vehicle and hasattr(vehicle, 'armed') and vehicle.armed:
                logger.info(f"Landing {name} drone...")
                try:
                    vehicle.mode = VehicleMode("LAND")
                except:
                    pass
        for vehicle, name in [(scout_vehicle, "Scout"), (treatment_vehicle, "Treatment")]:
            if vehicle:
                try:
                    vehicle.close()
                except:
                    pass
        try:
            rclpy.shutdown()
        except:
            pass
        logger.info("Instant confirmation system shutdown complete")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        logger.info("Received interrupt signal")
        stop_event.set()
    except Exception as e:
        logger.error(f"Fatal error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

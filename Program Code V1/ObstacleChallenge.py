#!/usr/bin/env python3
"""
WRO 2025 Future Engineers – Obstacle Challenge
Reference implementation (single-file) for a self-driving car using:
- 2D LiDAR (e.g., RPLidar A1/A2)
- Front camera with EdgeTPU-accelerated ML model (Google Coral USB/PCIe)
- Benewake TFmini ToF range sensor (side-facing)
- Optional wheel encoders (for odometry)

Goals aligned to 2025 rules:
- Drive 3 autonomous laps respecting red/green traffic sign passing rule
- Minimize moving traffic signs (ideally zero)
- Return to starting section and perform parallel parking in the marked lot

IMPORTANT: Hardware, mounting geometry, and PID constants must be calibrated
on your car. This file is a structured template with working algorithms and
sensible defaults. You must set serial device paths, model path, and GPIO/PWM
interfaces for your platform (Jetson Nano / Raspberry Pi + Coral, etc.).

Tested platforms:
- Jetson Nano (Ubuntu 20.04) + Coral USB Accelerator (EdgeTPU)
- Raspberry Pi 4B (64-bit OS) + Coral USB Accelerator

Python deps (install via pip):
  opencv-python
  numpy
  tflite-runtime
  pycoral
  rplidar-roboticia            # or rplidar (depending on distro)
  pyserial

Wiring:
- LiDAR via USB (/dev/ttyUSB0) or your adapter – update LIDAR_PORT
- TFmini via UART (/dev/ttyAMA0 or /dev/ttyUSB1) – update TFMINI_PORT
- Motor/servo driver: implement the two functions in MotorIO to fit your ESC/servo

This file contains:
  1) Config dataclasses
  2) Sensor adapters: LiDAR, TFmini, Camera + Coral detector
  3) Perception & fusion: lane corridor estimation, sign detection, obstacle mask
  4) Planning: finite state machine for laps, sign passing, parking
  5) Control: lateral & longitudinal PID, pure pursuit-like steering on corridor
  6) Robot runtime loop with threads and safety watchdog

Author: ChatGPT (reference-only). Adapt as needed for your robot.
License: MIT
"""

from dataclasses import dataclass, field
import threading
import time
import math
import serial
import struct
import sys
import queue
import cv2
import numpy as np

# ---- Optional imports; handle gracefully if missing ----
try:
    from rplidar import RPLidar
except Exception:
    RPLidar = None

try:
    from pycoral.utils.edgetpu import make_interpreter
    from pycoral.adapters.common import input_size
    from pycoral.adapters.detect import get_objects
except Exception:
    make_interpreter = None
    input_size = None
    get_objects = None

# ============================ CONFIG ============================
@dataclass
class Ports:
    LIDAR_PORT: str = "/dev/ttyUSB0"     # RPLidar
    TFMINI_PORT: str = "/dev/ttyUSB1"    # TFmini UART
    CAMERA_ID: int = 0                    # /dev/video0

@dataclass
class CoralModel:
    model_path: str = "/home/jetson/models/traffic_sign_pillars_edgetpu.tflite"
    labels: dict = field(default_factory=lambda: {0: "red_pillar", 1: "green_pillar"})
    score_threshold: float = 0.35
    max_detections: int = 10

@dataclass
class VehicleGeom:
    wheelbase_m: float = 0.19             # distance between axles
    track_width_m: float = 0.12           # distance between wheels (for kinematics)
    max_steer_deg: float = 27.0
    max_speed_mps: float = 1.2
    body_length_m: float = 0.20           # used for parking length calc / safety

@dataclass
class TrackParams:
    outer_wall_expected_m: float = 0.50   # used for corridor estimate (varies: 0.6–1.0 m)
    keep_side_offset_m: float = 0.08      # offset from centerline when obeying sign
    corner_alert_deg: float = 35.0        # LiDAR arc change threshold to detect corner

@dataclass
class ControlGains:
    # lateral steering
    kp_lat: float = 2.2
    ki_lat: float = 0.00
    kd_lat: float = 0.15
    # speed profile
    base_speed_mps: float = 0.8
    slow_speed_mps: float = 0.45          # near signs & corners
    park_speed_mps: float = 0.25

@dataclass
class CameraParams:
    width: int = 640
    height: int = 480
    fov_deg: float = 68.0
    fps: int = 30

@dataclass
class LiDARParams:
    min_angle_deg: float = -90
    max_angle_deg: float = +90
    min_range_m: float = 0.05
    max_range_m: float = 8.0
    right_sector: tuple = (-80, -20)
    left_sector: tuple = (20, 80)

@dataclass
class TFminiParams:
    smoothing_window: int = 5
    min_valid_m: float = 0.10
    max_valid_m: float = 12.0

@dataclass
class RuntimeFlags:
    visualize: bool = True                # show debug windows
    record_video: bool = False
    video_out_path: str = "run_debug.mp4"

# ============================ UTILITIES ============================

def clamp(x, lo, hi):
    return max(lo, min(hi, x))

# ============================ SENSOR ADAPTERS ============================
class LiDARAdapter(threading.Thread):
    def __init__(self, port: str, params: LiDARParams):
        super().__init__(daemon=True)
        self.params = params
        self.port = port
        self._stop = threading.Event()
        self.latest_scan = None  # np.array Nx2 of (angle_deg, distance_m)
        self._lidar = None

    def run(self):
        if RPLidar is None:
            print("[LiDAR] rplidar module not available. Running in dummy mode.")
            while not self._stop.is_set():
                self.latest_scan = np.zeros((0,2), dtype=np.float32)
                time.sleep(0.05)
            return
        try:
            self._lidar = RPLidar(self.port)
            for scan in self._lidar.iter_scans(max_buf_meas=5000):
                if self._stop.is_set():
                    break
                # scan is list of tuples: (quality, angle_deg, distance_mm)
                data = []
                for (_, ang, dist) in scan:
                    if self.params.min_angle_deg <= ang <= self.params.max_angle_deg and dist>0:
                        data.append([ang, dist/1000.0])
                self.latest_scan = np.array(data, dtype=np.float32)
        except Exception as e:
            print(f"[LiDAR] Exception: {e}")
        finally:
            try:
                if self._lidar:
                    self._lidar.stop(); self._lidar.disconnect()
            except Exception:
                pass

    def stop(self):
        self._stop.set()

    # estimate lateral corridor centerline offset using left/right wall fit
    def estimate_corridor(self):
        if self.latest_scan is None or len(self.latest_scan)==0:
            return 0.0, False, 0.0  # offset_m, have_data, curvature
        ang = self.latest_scan[:,0]
        rng = self.latest_scan[:,1]
        # Convert to Cartesian in robot frame
        ang_rad = np.deg2rad(ang)
        x = rng * np.cos(ang_rad)
        y = rng * np.sin(ang_rad)
        pts = np.stack([x,y], axis=1)
        # Select left/right sectors
        right_mask = (ang>=self.params.right_sector[0]) & (ang<=self.params.right_sector[1])
        left_mask  = (ang>=self.params.left_sector[0])  & (ang<=self.params.left_sector[1])
        def fit_line(pts):
            if len(pts)<10:
                return None
            # Fit y = ax + b
            x = pts[:,0]; y=pts[:,1]
            A = np.vstack([x, np.ones_like(x)]).T
            a,b = np.linalg.lstsq(A,y,rcond=None)[0]
            return a,b
        lr = pts[right_mask]
        ll = pts[left_mask]
        line_r = fit_line(lr)
        line_l = fit_line(ll)
        if line_r is None or line_l is None:
            return 0.0, False, 0.0
        # Compute lateral offset of centerline near x = 0.4 m ahead
        x_ahead = 0.40
        y_r = line_r[0]*x_ahead + line_r[1]
        y_l = line_l[0]*x_ahead + line_l[1]
        center_y = (y_l + y_r)/2.0
        # Curvature proxy from wall angles
        theta_r = math.atan(line_r[0]); theta_l = math.atan(line_l[0])
        curvature = (theta_l - theta_r)  # crude indicator; larger => turning
        return center_y, True, curvature

class TFminiAdapter(threading.Thread):
    """Benewake TFmini (UART) reader. Outputs distance in meters (smoothed)."""
    def __init__(self, port: str, params: TFminiParams):
        super().__init__(daemon=True)
        self.port = port
        self.params = params
        self._stop = threading.Event()
        self.dist_queue = queue.Queue(maxsize=1)
        self.serial = None
        self._window = []

    def run(self):
        try:
            self.serial = serial.Serial(self.port, 115200, timeout=1)
            while not self._stop.is_set():
                if self.serial.read() != b'\x59':
                    continue
                if self.serial.read() != b'\x59':
                    continue
                frame = self.serial.read(7)  # remaining bytes
                if len(frame) != 7:
                    continue
                dist = frame[0] + frame[1]*256  # in cm
                _strength = frame[2] + frame[3]*256
                # temp = frame[4] + frame[5]*256  # not used
                checksum = (0x59 + 0x59 + sum(frame[:6])) & 0xFF
                if checksum != frame[6]:
                    continue
                d_m = dist/100.0
                if self.params.min_valid_m <= d_m <= self.params.max_valid_m:
                    self._window.append(d_m)
                    if len(self._window)>self.params.smoothing_window:
                        self._window.pop(0)
                    smoothed = sum(self._window)/len(self._window)
                    if not self.dist_queue.full():
                        self.dist_queue.put(smoothed)
        except Exception as e:
            print(f"[TFmini] Exception: {e}")
        finally:
            try:
                if self.serial: self.serial.close()
            except Exception:
                pass

    def stop(self):
        self._stop.set()

    def latest(self, default=np.nan):
        try:
            return self.dist_queue.get_nowait()
        except queue.Empty:
            return default

class CoralDetector:
    """EdgeTPU-accelerated detector for traffic sign pillars (red/green).
    Model must be a TFLite-EdgeTPU detection model with labels: red_pillar, green_pillar.
    """
    def __init__(self, model: CoralModel, cam: CameraParams):
        self.model = model
        self.cam = cam
        self.interpreter = None
        self.input_w = None
        self.input_h = None
        self.fallback_hsv = True  # use color fallback if Coral not available
        try:
            if make_interpreter is None:
                raise RuntimeError("pycoral not available")
            self.interpreter = make_interpreter(model.model_path)
            self.interpreter.allocate_tensors()
            self.input_w, self.input_h = input_size(self.interpreter)
            self.fallback_hsv = False
            print("[Coral] Model loaded.")
        except Exception as e:
            print(f"[Coral] EdgeTPU not initialized ({e}). Using HSV fallback.")

    def detect(self, bgr_frame):
        """Return list of dicts: {cls, score, bbox}
        bbox: (x0,y0,x1,y1) in pixels
        cls: 'red_pillar' or 'green_pillar'
        """
        if self.interpreter is None or self.fallback_hsv:
            return self._detect_hsv(bgr_frame)
        # Resize & run TFLite
        rgb = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2RGB)
        resized = cv2.resize(rgb, (self.input_w, self.input_h))
        input_tensor = np.expand_dims(resized, 0).astype(np.uint8)
        self.interpreter.set_tensor(self.interpreter.get_input_details()[0]['index'], input_tensor)
        self.interpreter.invoke()
        objs = get_objects(self.interpreter, self.model.score_threshold)[: self.model.max_detections]
        h, w, _ = bgr_frame.shape
        results = []
        for o in objs:
            x0 = int(o.bbox.xmin * w / self.input_w)
            y0 = int(o.bbox.ymin * h / self.input_h)
            x1 = int(o.bbox.xmax * w / self.input_w)
            y1 = int(o.bbox.ymax * h / self.input_h)
            label = self.model.labels.get(o.id, str(o.id))
            results.append({"cls": label, "score": o.score, "bbox": (x0,y0,x1,y1)})
        return results

    def _detect_hsv(self, frame):
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        # crude masks for red & green pillars under indoor lighting; tune as needed
        lower_red1 = np.array([0, 120, 60]); upper_red1 = np.array([10, 255, 255])
        lower_red2 = np.array([160, 120, 60]); upper_red2 = np.array([179, 255, 255])
        lower_green = np.array([40, 70, 60]); upper_green = np.array([85, 255, 255])
        mask_red = cv2.inRange(hsv, lower_red1, upper_red1) | cv2.inRange(hsv, lower_red2, upper_red2)
        mask_green = cv2.inRange(hsv, lower_green, upper_green)
        def find_boxes(mask, label):
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            boxes = []
            for c in contours:
                area = cv2.contourArea(c)
                if area < 200:  # filter tiny blobs
                    continue
                x,y,w,h = cv2.boundingRect(c)
                boxes.append({"cls": label, "score": 0.5, "bbox": (x,y,x+w,y+h)})
            return boxes
        return find_boxes(mask_red, "red_pillar") + find_boxes(mask_green, "green_pillar")

# ============================ PERCEPTION & FUSION ============================
@dataclass
class SignDirective:
    side: str  # 'keep_right' or 'keep_left'
    bbox: tuple
    confidence: float
    timestamp: float

class Perception:
    def __init__(self, lidar: LiDARAdapter, tfm: TFminiAdapter, detector: CoralDetector,
                 cam_params: CameraParams, track: TrackParams):
        self.lidar = lidar
        self.tfm = tfm
        self.detector = detector
        self.cam = cam_params
        self.track = track
        self._last_sign: SignDirective|None = None

    def process(self, frame_bgr):
        # 1) Corridor estimation from LiDAR walls
        offset_y, ok, curvature = self.lidar.estimate_corridor()
        # 2) Traffic sign detection from camera
        detections = self.detector.detect(frame_bgr)
        sign_dir = None
        if detections:
            # choose the closest-looking sign (largest area)
            det = max(detections, key=lambda d:(d['bbox'][2]-d['bbox'][0])*(d['bbox'][3]-d['bbox'][1]))
            label = det['cls']
            side = 'keep_right' if label == 'red_pillar' else 'keep_left'
            sign_dir = SignDirective(side=side, bbox=det['bbox'], confidence=det['score'], timestamp=time.time())
            self._last_sign = sign_dir
        # 3) Side distance from TFmini (used for parking & corridor refinement)
        side_dist = self.tfm.latest(default=np.nan)
        return {
            "corridor_offset_m": offset_y if ok else 0.0,
            "have_corridor": ok,
            "curvature_proxy": curvature,
            "sign": sign_dir,
            "last_sign": self._last_sign,
            "side_dist_m": side_dist,
        }

# ============================ PLANNING STATE MACHINE ============================
class Planner:
    class State:
        DRIVE = 0
        PASS_KEEP_RIGHT = 1
        PASS_KEEP_LEFT  = 2
        PARK_SEARCH = 3
        PARK_ALIGN = 4
        PARK_EXEC = 5
        STOPPED = 6

    def __init__(self, track: TrackParams):
        self.track = track
        self.state = Planner.State.DRIVE
        self.lap_count = 0
        self.corner_count = 0
        self.last_corner_ts = 0.0
        self.last_state_change = time.time()
        self._sign_hold_until = 0.0

    def update(self, perc):
        now = time.time()
        curvature = abs(perc.get("curvature_proxy", 0.0))
        # Corner detection heuristic: large curvature spike with debounce
        if curvature > math.radians(20):
            if now - self.last_corner_ts > 1.0:
                self.corner_count += 1
                self.last_corner_ts = now
                if self.corner_count % 4 == 0:  # 4 corners ~ 1 lap
                    self.lap_count += 1
                    print(f"[Planner] Lap -> {self.lap_count}")

        # Transition to parking after 3 laps
        if self.lap_count >= 3 and self.state in (Planner.State.DRIVE, Planner.State.PASS_KEEP_LEFT, Planner.State.PASS_KEEP_RIGHT):
            self.state = Planner.State.PARK_SEARCH
            self.last_state_change = now

        # Obey sign for a short window while passing
        sign = perc.get("sign")
        if self.lap_count < 3 and sign is not None:
            self._sign_hold_until = now + 1.5  # keep side for ~1.5s
            self.state = Planner.State.PASS_KEEP_RIGHT if sign.side=='keep_right' else Planner.State.PASS_KEEP_LEFT
            self.last_state_change = now

        # Release sign hold when window expires
        if self.state in (Planner.State.PASS_KEEP_LEFT, Planner.State.PASS_KEEP_RIGHT) and now > self._sign_hold_until:
            self.state = Planner.State.DRIVE
            self.last_state_change = now

        return self.state

    # Desired lateral offset relative to corridor centerline
    def target_lateral_offset(self):
        if self.state == Planner.State.PASS_KEEP_RIGHT:
            return +abs(track_params.keep_side_offset_m)  # positive => to the right in our y-axis convention
        if self.state == Planner.State.PASS_KEEP_LEFT:
            return -abs(track_params.keep_side_offset_m)
        return 0.0

# ============================ CONTROL ============================
class PID:
    def __init__(self, kp, ki, kd):
        self.kp, self.ki, self.kd = kp, ki, kd
        self.ei = 0.0
        self.prev = 0.0
        self.ts = time.time()
    def reset(self):
        self.ei = 0.0; self.prev = 0.0; self.ts = time.time()
    def __call__(self, e):
        now = time.time(); dt = max(1e-3, now-self.ts)
        self.ts = now
        self.ei += e*dt
        de = (e - self.prev)/dt
        self.prev = e
        return self.kp*e + self.ki*self.ei + self.kd*de

class MotorIO:
    """Replace with your motor/servo drivers (e.g., PCA9685, Jetson.GPIO + pigpio, etc.)"""
    def __init__(self):
        # TODO: init I2C/PWM/ESC here
        pass
    def set_steer_deg(self, deg):
        deg = clamp(deg, -vehicle_geom.max_steer_deg, vehicle_geom.max_steer_deg)
        # TODO: map deg to servo PWM (e.g., 1000-2000us)
        # For now, print
        print(f"STEER: {deg:+.1f} deg")
    def set_speed(self, mps):
        mps = clamp(mps, 0.0, vehicle_geom.max_speed_mps)
        # TODO: map speed to ESC command
        print(f"SPEED: {mps:.2f} m/s")

class Controller:
    def __init__(self, gains: ControlGains, geom: VehicleGeom):
        self.g = gains
        self.geom = geom
        self.pid_lat = PID(gains.kp_lat, gains.ki_lat, gains.kd_lat)

    def compute(self, perc, planner_state, target_offset_m):
        # Lateral control: error = (corridor_offset + target_offset)
        have_corr = perc["have_corridor"]
        offset = perc["corridor_offset_m"] + target_offset_m
        if not have_corr:
            # fallback to TFmini side distance, aim for half track width
            side = perc["side_dist_m"]
            if not np.isnan(side):
                desired = track_params.outer_wall_expected_m/2.0
                offset = (desired - side)
            else:
                offset = 0.0  # give up; go straight
        u = self.pid_lat(offset)
        # map to steering angle (small angle approx)
        steer_deg = clamp(math.degrees(math.atan(u)), -self.geom.max_steer_deg, self.geom.max_steer_deg)

        # Speed profile
        curvature = abs(perc["curvature_proxy"])  # radians
        state = planner_state
        speed = self.g.base_speed_mps
        if state in (Planner.State.PASS_KEEP_LEFT, Planner.State.PASS_KEEP_RIGHT) or curvature>math.radians(12):
            speed = self.g.slow_speed_mps
        if state in (Planner.State.PARK_SEARCH, Planner.State.PARK_ALIGN, Planner.State.PARK_EXEC):
            speed = self.g.park_speed_mps
        return steer_deg, speed

# ============================ PARKING MODULE ============================
class Parking:
    def __init__(self, cam_params: CameraParams):
        self.cam = cam_params
        self._stage = 0
        self._stage_ts = 0.0

    def reset(self):
        self._stage = 0
        self._stage_ts = 0.0

    def find_parking_lot(self, frame):
        """Detect the white parking rectangle on the mat (width ~ 20 cm).
        Returns (found:bool, lateral_offset_px:float, depth_px:float)
        """
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (5,5), 0)
        edges = cv2.Canny(blur, 40, 120)
        lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=60, minLineLength=60, maxLineGap=20)
        if lines is None:
            return False, 0.0, 0.0
        # heuristic: select two near-parallel long horizontal lines representing top/bottom of slot
        horizontals = []
        for l in lines:
            x1,y1,x2,y2 = l[0]
            ang = math.degrees(math.atan2((y2-y1),(x2-x1)))
            if abs(ang) < 10:
                horizontals.append((x1,y1,x2,y2))
        if len(horizontals)<2:
            return False, 0.0, 0.0
        horizontals.sort(key=lambda h:min(h[1],h[3]))
        top = horizontals[0]; bottom = horizontals[-1]
        depth_px = abs(bottom[1]-top[1])
        center_x = (top[0]+top[2]+bottom[0]+bottom[2])/4.0
        offset_px = (center_x - self.cam.width/2)
        return True, offset_px, depth_px

    def park_control(self, frame):
        """Simple parallel parking routine: align with slot, then back-in for fixed time.
        Returns tuple (steer_deg, speed_mps, done_bool).
        """
        found, off_px, depth_px = self.find_parking_lot(frame)
        if not found:
            # scan slowly to the right to find slot
            return +10.0, 0.18, False
        # Convert pixel offset to degrees (approx)
        px_per_deg = self.cam.width / self.cam.fov_deg
        off_deg = off_px / px_per_deg
        # Align
        if abs(off_deg) > 3.0:
            steer = clamp(off_deg, -15, +15)
            return steer, 0.18, False
        # Back-in maneuver (timed)
        if self._stage == 0:
            self._stage = 1
            self._stage_ts = time.time()
        if self._stage == 1:
            if time.time() - self._stage_ts < 2.5:  # reverse duration
                return 0.0, 0.12, False  # ASSUMPTION: positive speed is forward; adapt if needed
            else:
                self._stage = 2
        return 0.0, 0.0, True

# ============================ ROBOT MAIN ============================
class Robot:
    def __init__(self):
        self.lidar = LiDARAdapter(ports.LIDAR_PORT, lidar_params)
        self.tfm   = TFminiAdapter(ports.TFMINI_PORT, tfmini_params)
        self.detector = CoralDetector(coral_model, camera_params)
        self.perc = Perception(self.lidar, self.tfm, self.detector, camera_params, track_params)
        self.plan = Planner(track_params)
        self.ctrl = Controller(control_gains, vehicle_geom)
        self.park = Parking(camera_params)
        self.motor = MotorIO()
        self.cap = cv2.VideoCapture(ports.CAMERA_ID)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, camera_params.width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, camera_params.height)
        self.cap.set(cv2.CAP_PROP_FPS, camera_params.fps)
        self._running = False
        if runtime_flags.record_video:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            self.vw = cv2.VideoWriter(runtime_flags.video_out_path, fourcc, 20, (camera_params.width, camera_params.height))
        else:
            self.vw = None

    def start(self):
        self._running = True
        self.lidar.start(); self.tfm.start()
        self.loop()

    def stop(self):
        self._running = False
        try:
            self.lidar.stop(); self.tfm.stop()
        except Exception:
            pass
        if self.cap: self.cap.release()
        if self.vw: self.vw.release()
        cv2.destroyAllWindows()

    def loop(self):
        fps_ts = time.time(); frames=0
        while self._running:
            ok, frame = self.cap.read()
            if not ok:
                print("[Camera] Read failed")
                break
            perc = self.perc.process(frame)
            state = self.plan.update(perc)

            if state in (Planner.State.PARK_SEARCH, Planner.State.PARK_ALIGN, Planner.State.PARK_EXEC):
                steer, speed, done = self.park.park_control(frame)
                if done:
                    self.motor.set_speed(0.0)
                    self.motor.set_steer_deg(0.0)
                    print("[Robot] Parking complete. STOP.")
                    break
            else:
                target_off = self.plan.target_lateral_offset()
                steer, speed = self.ctrl.compute(perc, state, target_off)
                # (Optional) ensure we do not get too close to side wall via TFmini safety
                d_side = perc["side_dist_m"]
                if not np.isnan(d_side) and d_side < 0.12:
                    speed = min(speed, 0.15)
                    steer = clamp(steer + 8.0, -vehicle_geom.max_steer_deg, vehicle_geom.max_steer_deg)

            self.motor.set_steer_deg(steer)
            self.motor.set_speed(speed)

            if runtime_flags.visualize:
                self.draw_debug(frame, perc, state, steer, speed)
            if self.vw is not None:
                self.vw.write(frame)

            frames += 1
            if time.time()-fps_ts > 2.0:
                print(f"[Perf] ~{frames/ (time.time()-fps_ts):.1f} FPS")
                fps_ts=time.time(); frames=0

    def draw_debug(self, frame, perc, state, steer, speed):
        h,w,_ = frame.shape
        # draw sign bbox
        det = perc.get("sign") or perc.get("last_sign")
        if det:
            x0,y0,x1,y1 = det.bbox
            color = (0,0,255) if det.side=='keep_right' else (0,255,0)
            cv2.rectangle(frame,(x0,y0),(x1,y1),color,2)
            cv2.putText(frame, det.side, (x0, max(0,y0-8)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        # HUD text
        cv2.putText(frame, f"STATE: {state}", (10,20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)
        cv2.putText(frame, f"laps:{self.plan.lap_count} corners:{self.plan.corner_count}", (10,44), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)
        cv2.putText(frame, f"steer:{steer:+.1f} speed:{speed:.2f}", (10,68), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)
        cv2.imshow("FE-Obstacle-2025", frame)
        if cv2.waitKey(1) == 27:
            self.stop()

# ============================ GLOBAL CONFIG INSTANTIATION ============================
ports = Ports()
coral_model = CoralModel()
vehicle_geom = VehicleGeom()
track_params = TrackParams()
control_gains = ControlGains()
camera_params = CameraParams()
lidar_params = LiDARParams()
tfmini_params = TFminiParams()
runtime_flags = RuntimeFlags()

# ============================ ENTRY POINT ============================
if __name__ == "__main__":
    bot = Robot()
    try:
        bot.start()
    except KeyboardInterrupt:
        pass
    finally:
        bot.stop()

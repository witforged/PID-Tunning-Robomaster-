# -*- coding:utf-8 -*-
import time
import csv
import threading
import cv2  # เปิดไว้ได้ จะใช้/ไม่ใช้ ขึ้นกับ SHOW_WINDOW
from robomaster import robot, vision

# ================== CONFIG ==================
CONN_TYPE = "ap"          
SHOW_WINDOW = True        # อยากดูภาพกับกรอบ marker ให้ True
FRAME_W, FRAME_H = 1280, 720
TARGET_X = 0.5            # ตรงกลางเฟรมแนวนอน (normalized)
TOL = 0.015               # เกณฑ์เข้าเป้า (normalized ~19 px ที่กว้าง 1280)
STEADY_NEED = 8           # ต้องนิ่งต่อเนื่องกี่เฟรม
MAX_TRACK_TIME = 5.0      # วินาที/marker
STEP_DURATION = 0.08      # วินาทีต่อ "ช่วงคำสั่ง" สำหรับคำนวณความเร็ว
MARKER_TTL = 0.30         # วินาที: อายุข้อมูล marker ที่จะถือว่ายัง "เห็นอยู่"
MAX_YAW_SPEED = 300.0     # deg/s กันเพดานความเร็ว

# ================== PID ==================
class PIDController:
    def __init__(self, Kp, Ki, Kd, i_limit=0.5, out_limit=6.0):
        self.Kp = Kp; self.Ki = Ki; self.Kd = Kd
        self.last_error = 0.0
        self.integral = 0.0
        self.i_limit = abs(i_limit)
        self.out_limit = abs(out_limit)

    def reset(self):
        self.last_error = 0.0
        self.integral = 0.0

    def update(self, error, dt):
        if dt <= 0:
            return 0.0
        self.integral += error * dt
        # anti-windup
        if self.integral > self.i_limit: self.integral = self.i_limit
        if self.integral < -self.i_limit: self.integral = -self.i_limit
        derivative = (error - self.last_error) / dt
        output = self.Kp * error + self.Ki * self.integral + self.Kd * derivative
        self.last_error = error
        # clamp output (ตีความเป็น "องศาต่อสเต็ป")
        if output > self.out_limit: output = self.out_limit
        if output < -self.out_limit: output = -self.out_limit
        return output

# ================== Marker Model ==================
class MarkerInfo:
    def __init__(self, x, y, w, h, info):
        self.x = x; self.y = y; self.w = w; self.h = h; self.info = info
        self.last_seen = time.time()
    @property
    def id(self): return str(self.info)
    @property
    def center_x_norm(self): return self.x
    @property
    def pt1(self):
        return int((self.x - self.w/2) * FRAME_W), int((self.y - self.h/2) * FRAME_H)
    @property
    def pt2(self):
        return int((self.x + self.w/2) * FRAME_W), int((self.y + self.h/2) * FRAME_H)
    @property
    def center(self):
        return int(self.x * FRAME_W), int(self.y * FRAME_H)
    @property
    def text(self): return self.id

# ================== Shared State ==================
_marker_lock = threading.Lock()
_marker_map = {}  # id -> MarkerInfo

def on_detect_marker(marker_info):
    """SDK ส่ง list ของ tuple: (x, y, w, h, info) — อัปเดตแบบสะสม + TTL"""
    global _marker_map
    now = time.time()
    if not marker_info:
        return
    with _marker_lock:
        for tup in marker_info:
            x, y, w, h, info = tup[0], tup[1], tup[2], tup[3], tup[4]
            key = str(info)  # แต่ละลายไม่ซ้ำ ใช้เป็น key ได้
            if key in _marker_map:
                m = _marker_map[key]
                m.x, m.y, m.w, m.h = x, y, w, h
                m.last_seen = now
            else:
                _marker_map[key] = MarkerInfo(x, y, w, h, info)
        # ลบตัวที่หายไปนานเกิน TTL
        dead = [k for k, v in _marker_map.items() if now - v.last_seen > MARKER_TTL]
        for k in dead:
            del _marker_map[k]

def get_markers_snapshot():
    now = time.time()
    with _marker_lock:
        return [m for m in _marker_map.values() if (now - m.last_seen) <= MARKER_TTL]

def order_LMR(markers):
    if len(markers) < 3: return []
    sorted_by_x = sorted(markers, key=lambda m: m.center_x_norm)
    return [sorted_by_x[0].id, sorted_by_x[1].id, sorted_by_x[2].id]

# ================== Optional Display Thread ==================
def display_loop(ep_camera, stop_flag):
    while not stop_flag["stop"]:
        img = ep_camera.read_cv2_image(strategy="newest", timeout=0.2)
        if img is None:
            continue
        ms = get_markers_snapshot()
        for m in ms:
            cv2.rectangle(img, m.pt1, m.pt2, (255, 255, 255), 2)
            cv2.putText(img, m.text, m.center, cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
        cv2.line(img, (int(TARGET_X*FRAME_W), 0), (int(TARGET_X*FRAME_W), FRAME_H), (200, 200, 200), 1)
        cv2.putText(img, f"count={len(ms)}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255,255,255), 2)
        cv2.imshow("Markers", img)
        if cv2.waitKey(1) & 0xFF == 27:
            break
    cv2.destroyAllWindows()

# ================== Main ==================
def main():
    ep_robot = robot.Robot()
    ep_robot.initialize(conn_type=CONN_TYPE)

    # >>> โหมดอิสระ: ให้กิมบอลหมุนเอง ล้อไม่ตาม <<<
    try:
        ep_robot.set_robot_mode(mode=robot.FREE)
    except Exception:
        pass

    ep_gimbal = ep_robot.gimbal
    ep_chassis = ep_robot.chassis
    ep_vision = ep_robot.vision
    ep_camera = ep_robot.camera

    # กันเหนียว: ล็อกล้อให้นิ่ง
    try:
        ep_chassis.drive_speed(x=0, y=0, z=0)
    except Exception:
        pass

    # เช็คเมธอดที่มีอยู่จริงใน SDK
    HAS_DRIVE_SPEED = hasattr(ep_gimbal, "drive_speed")
    HAS_MOVE = hasattr(ep_gimbal, "move")

    # เคลียร์ตำแหน่งกิมบอลก่อนเริ่ม
    try:
        ep_gimbal.moveto(pitch=0, yaw=0).wait_for_completed()
        time.sleep(0.2)
    except Exception:
        pass

    # เปิดวิดีโอสตรีม (เราแสดงภาพเองด้วย OpenCV -> display=False)
    ep_camera.start_video_stream(display=False)

    # สมัคร callback ตรวจจับ marker
    ep_vision.sub_detect_info(name="marker", callback=on_detect_marker)

    # ถ้าจะโชว์ภาพ ให้รันเธรดแสดงผล
    stop_flag = {"stop": False}
    if SHOW_WINDOW:
        disp_thread = threading.Thread(target=display_loop, args=(ep_camera, stop_flag), daemon=True)
        disp_thread.start()

    pid = PIDController(Kp=4.0, Ki=0.0, Kd=1.5, i_limit=0.6, out_limit=6.0)

    last_time = time.time()
    f = open("log_gimbal.csv", "w", newline="", encoding="utf-8")
    writer = csv.writer(f)
    writer.writerow(["Time", "MarkerID", "Error", "PID_Out(step_deg)", "Note"])

    try:
        while True:
            # กันเหนียว: รีเซ็ตความเร็วล้อเป็นศูนย์เรื่อย ๆ (บาง SDK มี inertia)
            try:
                ep_chassis.drive_speed(x=0, y=0, z=0)
            except Exception:
                pass

            ms = get_markers_snapshot()
            if len(ms) < 3:
                time.sleep(0.02)
                continue

            order = order_LMR(ms)
            if len(order) != 3:
                time.sleep(0.02)
                continue

            for target_id in order:
                pid.reset()
                steady = 0
                start = time.time()
                note = "tracking"

                while True:
                    now = time.time()
                    dt = now - last_time
                    last_time = now

                    # หา marker เป้าหมายล่าสุด
                    cur = None
                    for m in get_markers_snapshot():
                        if m.id == target_id:
                            cur = m
                            break

                    if cur is None:
                        note = "lost"
                        writer.writerow([now, target_id, "", "", note])
                        break

                    error = cur.center_x_norm - TARGET_X
                    out_deg = pid.update(error, dt if dt > 0 else 1e-3)  # องศาต่อ "สเต็ป"
                    # แปลงเป็นความเร็ว deg/s ตาม STEP_DURATION
                    yaw_speed_cmd = out_deg / STEP_DURATION
                    # clamp ความเร็ว
                    if yaw_speed_cmd > MAX_YAW_SPEED: yaw_speed_cmd = MAX_YAW_SPEED
                    if yaw_speed_cmd < -MAX_YAW_SPEED: yaw_speed_cmd = -MAX_YAW_SPEED

                    # ส่งคำสั่งกิมบอล (เฉพาะกิมบอล)
                    if HAS_DRIVE_SPEED:
                        # ทำ deadzone กันลื่น ถ้าเข้าเกณฑ์แล้วให้สั่ง 0 ไปเลย
                        cmd = 0.0 if abs(error) < TOL else yaw_speed_cmd
                        ep_gimbal.drive_speed(pitch_speed=0, yaw_speed=cmd)

                    elif HAS_MOVE:
                        ep_gimbal.move(yaw=-out_deg, pitch=0, yaw_speed=180, pitch_speed=180)
                    else:
                        print("Gimbal API not found: neither drive_speed nor move")
                        note = "api_missing"
                        writer.writerow([now, target_id, error, out_deg, note])
                        break

                    writer.writerow([now, target_id, error, out_deg, note])

                    # เช็คเข้าเป้าแบบนิ่ง
                    if abs(error) < TOL:
                        steady += 1
                    else:
                        steady = 0
                    if steady >= STEADY_NEED:
                        note = "locked"
                        writer.writerow([time.time(), target_id, error, out_deg, note])
                        break

                    # กันลูปไม่จบ
                    if (time.time() - start) > MAX_TRACK_TIME:
                        note = "timeout"
                        writer.writerow([time.time(), target_id, error, out_deg, note])
                        break

                    time.sleep(0.02)

            time.sleep(0.5)  # พักก่อนเริ่มรอบใหม่

    except KeyboardInterrupt:
        pass
    finally:
        try: ep_vision.unsub_detect_info(name="marker")
        except Exception: pass
        try: stop_flag["stop"] = True
        except Exception: pass
        try: ep_camera.stop_video_stream()
        except Exception: pass
        # หยุดหมุนกิมบอล
        try:
            if hasattr(ep_robot.gimbal, "drive_speed"):
                ep_robot.gimbal.drive_speed(pitch_speed=0, yaw_speed=0)
        except Exception:
            pass
        # หยุดล้อ
        try:
            ep_chassis.drive_speed(x=0, y=0, z=0)
        except Exception:
            pass
        f.close()
        ep_robot.close()

if __name__ == "__main__":
    main()

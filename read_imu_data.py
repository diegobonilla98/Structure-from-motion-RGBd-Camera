import serial
import threading
import time
from datetime import datetime, timezone
from collections import deque
import math


class IMUSerialReader(threading.Thread):
    def __init__(self, port='COM3', baud_rate=9600, buffer_size=1000):
        super().__init__(daemon=True)
        self.port = port
        self.baud_rate = baud_rate
        self.buffer = deque(maxlen=buffer_size)
        self.buffer_lock = threading.Lock()
        self.running = True

        # Conversion constants
        self.ACCEL_SCALE = 2 / 32768  # g per LSB
        self.GYRO_SCALE = 250 / 32768  # deg/s per LSB
        self.G_CONST = 9.80111  # m/s² per g

        # Calibration values
        self.calibrated = False
        self.accel_bias_raw = (0, 0, 0)
        self.gyro_bias_raw = (0, 0, 0)

        try:
            self.ser = serial.Serial(self.port, self.baud_rate, timeout=1)
            time.sleep(2)  # Allow serial to stabilize
        except serial.SerialException as e:
            raise RuntimeError(f"Could not open serial port {self.port}: {e}")

    def run(self):
        print(f"[IMUSerialReader] Started reading from {self.port}")
        while self.running:
            try:
                if self.ser.in_waiting:
                    line = self.ser.readline().decode('utf-8').strip()
                    timestamp = datetime.now(timezone.utc)
                    raw = self.parse_raw(line)
                    if raw:
                        if not self.calibrated:
                            self.gyro_bias_raw = raw[3:]  # Only gyro bias
                            self.calibrated = True
                            print(f"[IMUSerialReader] Gyro calibration complete: gyro_bias={self.gyro_bias_raw}")
                            continue
                        parsed = self.convert_to_si(raw)
                        with self.buffer_lock:
                            self.buffer.append((timestamp, parsed))
            except Exception as e:
                print(f"[IMUSerialReader] Error: {e}")

    def parse_raw(self, line):
        try:
            values = list(map(int, line.split(',')))
            if len(values) != 6:
                return None
            return tuple(values)  # ax, ay, az, gx, gy, gz
        except:
            return None

    def convert_to_si(self, raw_values):
        ax, ay, az, gx, gy, gz = raw_values
        bgx, bgy, bgz = self.gyro_bias_raw  # Only apply to gyro

        # Keep accel values as-is
        # Remove gyro bias
        gx -= bgx
        gy -= bgy
        gz -= bgz

        # Convert acceleration to m/s²
        ax_mps2 = ax * self.ACCEL_SCALE * self.G_CONST
        ay_mps2 = ay * self.ACCEL_SCALE * self.G_CONST
        az_mps2 = az * self.ACCEL_SCALE * self.G_CONST

        # Convert gyro to rad/s
        gx_rads = gx * self.GYRO_SCALE * (math.pi / 180)
        gy_rads = gy * self.GYRO_SCALE * (math.pi / 180)
        gz_rads = gz * self.GYRO_SCALE * (math.pi / 180)

        return {
            "accel_mps2": (ax_mps2, ay_mps2, az_mps2),
            "gyro_rads": (gx_rads, gy_rads, gz_rads),
            "accel_total_mps2": math.sqrt(ax_mps2**2 + ay_mps2**2 + az_mps2**2)
        }

    def get_closest_reading(self, target_timestamp: datetime):
        with self.buffer_lock:
            if not self.buffer:
                return None
            closest = min(self.buffer, key=lambda x: abs((x[0] - target_timestamp).total_seconds()))
            return {
                "timestamp": closest[0].isoformat(),
                "accel_mps2": closest[1]["accel_mps2"],
                "gyro_rads": closest[1]["gyro_rads"],
                "accel_total_mps2": closest[1]["accel_total_mps2"]
            }

    def stop(self):
        self.running = False
        if self.ser and self.ser.is_open:
            self.ser.close()


if __name__ == "__main__":
    imu_reader = IMUSerialReader(port='COM3')
    imu_reader.start()

    try:
        while True:
            cmd = input("Enter ISO timestamp or press Enter for now:\n> ").strip()
            if cmd == "":
                ts = datetime.now(timezone.utc)
            else:
                try:
                    ts = datetime.fromisoformat(cmd).astimezone(timezone.utc)
                except:
                    print("Invalid timestamp.")
                    continue

            result = imu_reader.get_closest_reading(ts)
            if result:
                print(result)
            else:
                print("No data yet.")

    except KeyboardInterrupt:
        print("Stopping...")
        imu_reader.stop()


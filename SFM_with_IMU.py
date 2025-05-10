import signal
import sys
import cv2
import numpy as np
import open3d as o3d
import pyrealsense2 as rs
from scipy.spatial.transform import Rotation as R
from datetime import datetime, timezone
import time

from read_imu_data import IMUSerialReader

def sigint_handler(signum, frame):
    global imu_reader
    print("\nStopping…")
    if imu_reader is not None and imu_reader.running:
        print("Stopping IMU reader...")
        imu_reader.stop()
        imu_reader.join()
    sys.exit(0)

def depth_to_points(depth_np, depth_scale, intr, d_min, d_max):
    height, width = depth_np.shape
    x, y = np.meshgrid(np.arange(width), np.arange(height))
    z = depth_np * depth_scale

    # Filter out zero depth and outliers
    mask = (z > d_min) & (z < d_max)
    border = 5
    mask &= (x > border) & (x < width - border) & (y > border) & (y < height - border)

    x_valid = x[mask]
    y_valid = y[mask]
    z_valid = z[mask]

    # Camera intrinsics
    fx, fy = intr.fx, intr.fy
    cx, cy = intr.ppx, intr.ppy

    # Back-project to 3D
    X = (x_valid - cx) * z_valid / fx
    Y = (y_valid - cy) * z_valid / fy
    Z = z_valid

    points = np.stack((X, Y, Z), axis=-1)
    return points

def extract_info_from_dT(dT):
    translation = dT[:3, 3]
    rotation_matrix = dT[:3, :3]
    translation_total = np.linalg.norm(translation)
    r = R.from_matrix(rotation_matrix)
    roll, pitch, yaw = r.as_euler('xyz', degrees=False)
    angle_axis, _ = cv2.Rodrigues(rotation_matrix)
    rotation_total = np.linalg.norm(angle_axis)
    return translation, rotation_matrix, translation_total, roll, pitch, yaw, rotation_total

signal.signal(signal.SIGINT, sigint_handler)

imu_reader = None
try:
    imu_reader = IMUSerialReader(port='COM3')
    imu_reader.start()
    print("IMU Reader started. Waiting for calibration...")
    calibration_wait_start = time.time()
    while imu_reader.running and not imu_reader.calibrated and (time.time() - calibration_wait_start < 5):
        time.sleep(0.1)
    if imu_reader.calibrated:
        print("IMU calibrated.")
    elif not imu_reader.running:
        print("IMU reader did not start/stay running correctly.")
        imu_reader = None
    else:
        print("IMU calibration timed out. Proceeding without IMU-based initial guess.")
except Exception as e:
    print(f"Failed to initialize IMU reader: {e}")
    imu_reader = None

vis = o3d.visualization.Visualizer()
vis.create_window(window_name="Real-time RGB-D camera pose estimation")

view_ctl = vis.get_view_control()
view_ctl.set_lookat([0, 0, 0])
view_ctl.set_front([0, 0, -1])
view_ctl.set_up([0, -1, 0])

pipeline = rs.pipeline()
cfg = rs.config()
cfg.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
cfg.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

spatial = rs.spatial_filter()
spatial.set_option(rs.option.filter_magnitude, 2)
spatial.set_option(rs.option.filter_smooth_alpha, 0.5)
spatial.set_option(rs.option.filter_smooth_delta, 20)
spatial.set_option(rs.option.holes_fill, 5)

profile = pipeline.start(cfg)
align_to_color = rs.align(rs.stream.color)

depth_sensor = profile.get_device().first_depth_sensor()
depth_scale = depth_sensor.get_depth_scale()
depth_sensor.set_option(rs.option.visual_preset, 1)

intr = profile.get_stream(rs.stream.color).as_video_stream_profile().get_intrinsics()
cam_intr = o3d.camera.PinholeCameraIntrinsic(intr.width, intr.height, intr.fx, intr.fy, intr.ppx, intr.ppy)

prev_rgbd = None
T_global = np.eye(4)
prev_frame_dt = None

first_frame = True

MAX_DEPTH = 6.0
MIN_DEPTH = 0.2

option = o3d.pipelines.odometry.OdometryOption()
option.depth_diff_max = 0.07
option.depth_min = MIN_DEPTH
option.depth_max = MAX_DEPTH

global_pcd = o3d.geometry.PointCloud()
global_point_cloud = None

cv2.namedWindow("Debug Logs", cv2.WINDOW_AUTOSIZE)
cv2.namedWindow("Depth Image", cv2.WINDOW_AUTOSIZE)
cv2.namedWindow("Color Image", cv2.WINDOW_AUTOSIZE)

try:
    while True:
        frames = pipeline.wait_for_frames()
        frames = align_to_color.process(frames)

        # Use system time for timestamp
        current_frame_dt = datetime.now(timezone.utc)

        depth_frame = frames.get_depth_frame()
        color_frame = frames.get_color_frame()
        if not depth_frame or not color_frame:
            continue
        depth_frame = spatial.process(depth_frame)

        depth_np = np.asanyarray(depth_frame.get_data()).copy()
        color_np = np.asanyarray(color_frame.get_data()).copy()
        color_np = cv2.cvtColor(color_np, cv2.COLOR_BGR2RGB)

        points = depth_to_points(depth_np, depth_scale, intr, MIN_DEPTH, MAX_DEPTH)

        if prev_rgbd is None:
            global_point_cloud = points
            global_pcd.points = o3d.utility.Vector3dVector(global_point_cloud)
            if len(points) == depth_np.shape[0] * depth_np.shape[1]:
                global_pcd.colors = o3d.utility.Vector3dVector(color_np.reshape(-1, 3) / 255.0)

        o3d_depth = o3d.geometry.Image(depth_np)
        o3d_color = o3d.geometry.Image(color_np)

        rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
            o3d_color,
            o3d_depth,
            depth_scale=1.0 / depth_scale,
            depth_trunc=MAX_DEPTH,
            convert_rgb_to_intensity=False
        )

        init_transform_guess = np.identity(4)
        if prev_rgbd is not None and prev_frame_dt is not None and imu_reader is not None and imu_reader.calibrated:
            time_delta_seconds = (current_frame_dt - prev_frame_dt).total_seconds()
            if time_delta_seconds > 1e-6:
                imu_data = imu_reader.get_closest_reading(current_frame_dt)
                if imu_data and 'gyro_rads' in imu_data:
                    gyro_rads = np.array(imu_data["gyro_rads"])
                    rotation_vector = gyro_rads * time_delta_seconds
                    try:
                        R_from_imu = R.from_rotvec(rotation_vector).as_matrix()
                        init_transform_guess[:3, :3] = R_from_imu
                    except Exception as e:
                        print(f"Error creating rotation from IMU data: {e}")

        if prev_rgbd is not None:
            success, dT, info = o3d.pipelines.odometry.compute_rgbd_odometry(
                prev_rgbd, rgbd, cam_intr,
                init_transform_guess,
                o3d.pipelines.odometry.RGBDOdometryJacobianFromHybridTerm(),
                option
            )
            if success:
                T_global = T_global @ dT

                # Calculate per-frame movement (dT)
                translation = T_global[:3, 3]
                rotation_matrix = T_global[:3, :3]

                # Inverse the transformation to get the points in the global frame
                T_global_inv = np.linalg.inv(T_global)
                points_transformed = np.dot(points, T_global_inv[:3, :3].T) + T_global_inv[:3, 3]

                global_point_cloud = np.vstack((global_point_cloud, points_transformed))
                temp_pcd = o3d.geometry.PointCloud()
                temp_pcd.points = o3d.utility.Vector3dVector(global_point_cloud)
                temp_pcd = temp_pcd.voxel_down_sample(voxel_size=0.1)
                global_point_cloud = np.asarray(temp_pcd.points)
                global_pcd.points = o3d.utility.Vector3dVector(global_point_cloud)

                translation, rotation_matrix, translation_total, roll, pitch, yaw, rotation_total = extract_info_from_dT(dT)
                global_translation, global_rotation_matrix, global_translation_total, global_roll, global_pitch, global_yaw, global_rotation_total = extract_info_from_dT(T_global)

                debug_text = (
                    f"Frame movement: Δx={translation[0]:.4f} Δy={translation[1]:.4f} Δz={translation[2]:.4f} (total={translation_total:.4f})\n"
                    f"Frame rotation: yaw={yaw:.4f} pitch={pitch:.4f} roll={roll:.4f} (total={rotation_total:.4f})\n"
                    f"Global movement: Δx={global_translation[0]:.4f} Δy={global_translation[1]:.4f} Δz={global_translation[2]:.4f} (total={global_translation_total:.4f})\n"
                    f"Global rotation: yaw={global_yaw:.4f} pitch={global_pitch:.4f} roll={global_roll:.4f} (total={global_rotation_total:.4f})\n"
                    f"Number of points in global point cloud: {len(global_point_cloud)}\n"
                )
                debug_img = np.zeros((200, 1000, 3), dtype=np.uint8)
                y0, dy = 30, 30
                for i, line in enumerate(debug_text.split('\n')):
                    cv2.putText(debug_img, line, (10, y0 + i * dy), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                cv2.imshow("Debug Logs", debug_img)

                camera_pose_visualizer = o3d.geometry.TriangleMesh.create_coordinate_frame(size=10.)
                camera_pose_visualizer.transform(T_global)
                vis.update_geometry(camera_pose_visualizer)

        if first_frame:
            first_frame = False
            vis.add_geometry(global_pcd)
        vis.update_geometry(global_pcd)

        vis.poll_events()
        vis.update_renderer()

        if not vis.poll_events():
            break

        prev_rgbd = rgbd
        prev_frame_dt = current_frame_dt

        depth_cv = cv2.applyColorMap(cv2.convertScaleAbs(depth_np, alpha=0.03), cv2.COLORMAP_JET)
        color_cv = cv2.cvtColor(color_np, cv2.COLOR_RGB2BGR)
        cv2.imshow("Depth Image", depth_cv)
        cv2.imshow("Color Image", color_cv)
        key = cv2.waitKey(1)
        if key == 27 or key == ord('q'):
            break
finally:
    if imu_reader is not None and imu_reader.running:
        print("Stopping IMU reader in finally block...")
        imu_reader.stop()
        imu_reader.join()
    vis.destroy_window()
    pipeline.stop()
    cv2.destroyAllWindows()
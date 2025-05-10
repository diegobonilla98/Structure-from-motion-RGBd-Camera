import signal
import sys
import cv2
import numpy as np
import open3d as o3d
import pyrealsense2 as rs
from scipy.spatial.transform import Rotation as R



def sigint_handler(signum, frame):
    print("\nStopping…")
    sys.exit(0)


signal.signal(signal.SIGINT, sigint_handler)

vis = o3d.visualization.Visualizer()
vis.create_window(window_name="Real-time RGB-D camera pose estimation")
camera_pose_visualizer = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1)
vis.add_geometry(camera_pose_visualizer)

view_ctl = vis.get_view_control()
view_ctl.set_zoom(10)

pipeline = rs.pipeline()
cfg = rs.config()
cfg.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
cfg.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

spatial = rs.spatial_filter()
spatial.set_option(rs.option.filter_magnitude, 2)  # Number of filter iterations
spatial.set_option(rs.option.filter_smooth_alpha, 0.5) # Smoothing factor
spatial.set_option(rs.option.filter_smooth_delta, 20) # Step size boundary
spatial.set_option(rs.option.holes_fill, 5) # 0 = disabled, 1-5 = various hole filling algorithms

profile = pipeline.start(cfg)

align_to_color = rs.align(rs.stream.color)

depth_sensor = profile.get_device().first_depth_sensor()
depth_scale = depth_sensor.get_depth_scale()
print(f"Depth scale: {depth_scale:.6f} m per LSB")

intr = profile.get_stream(rs.stream.color).as_video_stream_profile().get_intrinsics()
cam_intr = o3d.camera.PinholeCameraIntrinsic(intr.width, intr.height, intr.fx, intr.fy, intr.ppx, intr.ppy)

prev_rgbd = None
T_global = np.eye(4)

first_frame = True

option = o3d.pipelines.odometry.OdometryOption()
option.depth_diff_max = 0.07
option.depth_min = 0.2
option.depth_max = 3.0

while True:
    frames = pipeline.wait_for_frames()
    frames = align_to_color.process(frames)

    depth_frame = frames.get_depth_frame()
    color_frame = frames.get_color_frame()
    if not depth_frame or not color_frame:
        continue
    
    depth_frame = spatial.process(depth_frame)

    depth_np = np.asanyarray(depth_frame.get_data()).copy()
    color_np = np.asanyarray(color_frame.get_data()).copy()

    o3d_depth = o3d.geometry.Image(depth_np)
    o3d_color = o3d.geometry.Image(color_np)
    o3d_color = o3d.geometry.Image(np.ascontiguousarray(color_np[:, :, ::-1]))

    rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
        o3d_color,
        o3d_depth,
        depth_scale=1.0 / depth_scale,
        depth_trunc=3.0,
        convert_rgb_to_intensity=False
    )

    if prev_rgbd is not None:
        success, dT, info = o3d.pipelines.odometry.compute_rgbd_odometry(
            prev_rgbd, rgbd, cam_intr,
            np.identity(4),
            o3d.pipelines.odometry.RGBDOdometryJacobianFromHybridTerm(),
            option
        )
        if success:
            T_global = T_global @ dT

            # Calculate per-frame movement (dT)
            translation = dT[:3, 3]
            rotation_matrix = dT[:3, :3]

            # Total translation (Euclidean norm)
            translation_total = np.linalg.norm(translation)
            # Euler angles from rotation matrix (in radians)
            r = R.from_matrix(rotation_matrix)
            roll, pitch, yaw = r.as_euler('xyz', degrees=False)
            # Total rotation (angle of rotation)
            angle_axis, _ = cv2.Rodrigues(rotation_matrix)
            rotation_total = np.linalg.norm(angle_axis)

            print(f"Frame movement: Δx={translation[0]:.4f} Δy={translation[1]:.4f} Δz={translation[2]:.4f} (total={translation_total:.4f})")
            print(f"Frame rotation: yaw={yaw:.4f} pitch={pitch:.4f} roll={roll:.4f} (total={rotation_total:.4f})")

            # Calculate global movement (T_global)
            global_translation = T_global[:3, 3]
            global_translation_total = np.linalg.norm(global_translation)
            global_rotation_matrix = T_global[:3, :3]
            r_global = R.from_matrix(global_rotation_matrix)
            global_roll, global_pitch, global_yaw = r_global.as_euler('xyz', degrees=False)
            global_angle_axis, _ = cv2.Rodrigues(global_rotation_matrix)
            global_rotation_total = np.linalg.norm(global_angle_axis)

            print(f"Global movement: Δx={global_translation[0]:.4f} Δy={global_translation[1]:.4f} Δz={global_translation[2]:.4f} (total={global_translation_total:.4f})")
            print(f"Global rotation: yaw={global_yaw:.4f} pitch={global_pitch:.4f} roll={global_roll:.4f} (total={global_rotation_total:.4f})")

            print("-" * 50)

            vis.remove_geometry(camera_pose_visualizer, reset_bounding_box=False)
            camera_pose_visualizer = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1)
            camera_pose_visualizer.transform(T_global)
            vis.add_geometry(camera_pose_visualizer, reset_bounding_box=False)

    if first_frame:
        first_frame = False
    vis.poll_events()
    vis.update_renderer()

    if not vis.poll_events():
        break

    prev_rgbd = rgbd


vis.destroy_window()
pipeline.stop()

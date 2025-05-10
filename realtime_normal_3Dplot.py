import pyrealsense2 as rs
import numpy as np
import open3d as o3d
import cv2


# Configure depth and color streams
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

# Start streaming
profile = pipeline.start(config)
depth_sensor = profile.get_device().first_depth_sensor()
depth_scale = depth_sensor.get_depth_scale()

# Initialize RealSense spatial filter
spatial = rs.spatial_filter()
spatial.set_option(rs.option.filter_magnitude, 2)  # Number of filter iterations
spatial.set_option(rs.option.filter_smooth_alpha, 0.5) # Smoothing factor
spatial.set_option(rs.option.filter_smooth_delta, 20) # Step size boundary
spatial.set_option(rs.option.holes_fill, 5) # 0 = disabled, 1-5 = various hole filling algorithm

depth_sensor.set_option(rs.option.visual_preset, 6)

# Create Open3D visualizer
vis = o3d.visualization.Visualizer()
vis.create_window("RealSense D415 Real-time Point Cloud")

pcd = o3d.geometry.PointCloud()
added = False

cv2.namedWindow("Depth Image", cv2.WINDOW_AUTOSIZE)
cv2.namedWindow("Color Image", cv2.WINDOW_AUTOSIZE)

while True:
    # Wait for a coherent pair of frames: depth and color
    frames = pipeline.wait_for_frames()
    depth_frame = frames.get_depth_frame()
    color_frame = frames.get_color_frame()
    if not depth_frame or not color_frame:
        continue

    # Apply RealSense post-processing filter to the depth frame
    depth_frame = spatial.process(depth_frame)

    # Convert images to numpy arrays
    depth_image = np.asanyarray(depth_frame.get_data())
    color_image = np.asanyarray(color_frame.get_data())

    # Convert BGR to RGB
    color_image = cv2.cvtColor(color_image, cv2.COLOR_BGR2RGB)

    # Intrinsics & point cloud calculation
    intrinsics = depth_frame.profile.as_video_stream_profile().intrinsics
    fx, fy = intrinsics.fx, intrinsics.fy
    cx, cy = intrinsics.ppx, intrinsics.ppy
    height, width = depth_image.shape

    # Create a grid of pixel coordinates
    x, y = np.meshgrid(np.arange(width), np.arange(height))
    z = depth_image * depth_scale

    # # Filter out zero depth and outliers
    mask = (z > 0.2) & (z < 6.0)
    border = 5
    mask &= (x > border) & (x < width - border) & (y > border) & (y < height - border)

    x = x[mask]
    y = y[mask]
    z = z[mask]

    # Back-project to 3D
    X = (x - cx) * z / fx
    Y = (y - cy) * z / fy
    Z = z

    points = np.stack((X, Y, Z), axis=-1)

    # Get corresponding colors
    colors = color_image[y, x] / 255.0

    # Update Open3D point cloud
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.colors = o3d.utility.Vector3dVector(colors)

    if not added:
        vis.add_geometry(pcd)
        added = True

    vis.update_geometry(pcd)
    vis.poll_events()
    vis.update_renderer()

    if not vis.poll_events():
        break

    # Display images
    depth_cv = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET)
    color_cv = cv2.cvtColor(color_image, cv2.COLOR_RGB2BGR)
    cv2.imshow("Depth Image", depth_cv)
    cv2.imshow("Color Image", color_cv)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

pipeline.stop()
vis.destroy_window()

# Real-time RGB-D SLAM with Intel RealSense

This project implements a real-time Structure from Motion (SfM) / SLAM (Simultaneous Localization and Mapping) system using an Intel RealSense D400 series camera. It captures RGB and depth data, performs visual odometry to estimate camera pose, and reconstructs a 3D point cloud of the environment. This version utilizes OpenCV's `RgbdOdometry` for pose estimation.

## Features

*   **Real-time Camera Pose Estimation**: Tracks the 6DoF (Degrees of Freedom) pose of the RealSense camera.
*   **3D Point Cloud Reconstruction**: Builds a global 3D point cloud map of the scanned environment.
*   **Visualization**:
    *   Live 3D point cloud visualization using Open3D.
    *   Displays the current camera pose as a coordinate frame.
    *   Shows live depth and color image feeds using OpenCV.
    *   Debug window with real-time translation and rotation information (frame-to-frame and global).
*   **RealSense Configuration**:
    *   Configures depth and color streams.
    *   Applies a spatial filter to the depth data for noise reduction.
    *   Aligns depth frames to the color frame.
*   **Interactive Controls**:
    *   Press 's' in the 3D visualizer window or the image windows to pause/resume the mapping and pose estimation process.
    *   Press 'q' or 'ESC' in the image windows to quit.

## Dependencies

*   Python 3.x
*   OpenCV (`opencv-python` and `opencv-contrib-python` for `cv2.rgbd`)
*   NumPy (`numpy`)
*   SciPy (`scipy`) for rotation conversions.
*   Open3D (`open3d`) for 3D visualization and point cloud processing.
*   PyRealSense2 (`pyrealsense2`) for Intel RealSense camera interface.

You can typically install these using pip:
```bash
pip install opencv-python opencv-contrib-python numpy scipy open3d pyrealsense2
```

## How to Run

1.  **Connect your Intel RealSense camera.**
2.  **Ensure all dependencies are installed.**
3.  **Execute the script:**
    ```bash
    python better_SFM.py
    ```

## Script Overview (`better_SFM.py`)

*   **Initialization**:
    *   Sets up signal handling for graceful exit (Ctrl+C).
    *   Initializes Open3D visualizer and registers key callbacks.
    *   Configures and starts the RealSense pipeline (depth and color streams).
    *   Applies a spatial filter to the depth stream.
    *   Retrieves camera intrinsics.
    *   Initializes OpenCV's `RgbdOdometry`.
*   **Main Loop**:
    *   Waits for and aligns RealSense frames.
    *   Processes depth and color frames:
        *   Converts frames to NumPy arrays.
        *   Converts depth data to a 3D point cloud.
    *   **Odometry**:
        *   If it's not the first frame and not paused, it computes the transformation (dT) between the current and previous RGB-D frames using `cv2.rgbd.RgbdOdometry.compute()`.
        *   Updates the global camera pose (`T_global`).
        *   Transforms the current frame's points into the global coordinate system.
        *   Appends the transformed points to the global point cloud.
        *   Downsamples the global point cloud using `voxel_down_sample` to manage density.
    *   **Visualization**:
        *   Updates the Open3D visualizer with the global point cloud and the current camera pose.
        *   Displays debug information (translation, rotation) in an OpenCV window.
        *   Shows live depth and color images.
    *   Handles user input for pausing/resuming or quitting.
*   **Cleanup**:
    *   Stops the RealSense pipeline and closes all OpenCV and Open3D windows upon exit.

## Key Functions

*   [`sigint_handler`](g%3A%2FMy%20Drive%2FPythonProjects%2FSfMRealsense%2Fbetter_SFM.py): Handles interrupt signals for a clean shutdown.
*   [`stop_callback`](g%3A%2FMy%20Drive%2FPythonProjects%2FSfMRealsense%2Fbetter_SFM.py): Toggles the `stopped` flag when 's' is pressed.
*   [`depth_to_points`](g%3A%2FMy%20Drive%2FPythonProjects%2FSfMRealsense%2Fbetter_SFM.py): Converts a depth frame into a 3D point cloud using camera intrinsics.
*   [`extract_info_from_dT`](g%3A%2FMy%20Drive%2FPythonProjects%2FSfMRealsense%2Fbetter_SFM.py): Extracts translation, rotation matrix, Euler angles, and total movement from a transformation matrix.
*   [`rgbd_dt`](g%3A%2FMy%20Drive%2FPythonProjects%2FSfMRealsense%2Fbetter_SFM.py): A wrapper around `cv2.rgbd.RgbdOdometry_create().compute()` to estimate the relative pose between two RGB-D frames.

## Configuration Constants

*   `MAX_DEPTH`: Maximum depth value (in meters) to consider for point cloud generation.
*   `MIN_DEPTH`: Minimum depth value (in meters) to consider.
*   `voxel_size` (inside the main loop, for `temp_pcd.voxel_down_sample`): Controls the resolution of the downsampled global point cloud. A smaller value means a denser cloud.

This script provides a robust foundation for RGB-D SLAM experiments and can be extended further with features like loop closure, IMU integration (see [SFM_with_IMU.py](g%3A%2FMy%20Drive%2FPythonProjects%2FSfMRealsense%2FSFM_with_IMU.py)), or more advanced odometry techniques.
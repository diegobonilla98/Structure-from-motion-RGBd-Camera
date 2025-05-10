import pyrealsense2 as rs
import numpy as np
import cv2

def main():
    print("🔍 Starting RealSense D415 Full Diagnostic with OpenCV...")

    # Create pipeline and config
    pipeline = rs.pipeline()
    config = rs.config()

    try:
        # Enable streams
        config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
        config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
        config.enable_stream(rs.stream.infrared, 1, 640, 480, rs.format.y8, 30)

        # Start streaming
        profile = pipeline.start(config)
        print("✅ Pipeline started successfully.")

        # Main loop
        while True:
            try:
                frames = pipeline.wait_for_frames(timeout_ms=5000)
            except Exception as e:
                print("❌ Frame wait timeout:", str(e))
                break

            depth_frame = frames.get_depth_frame()
            color_frame = frames.get_color_frame()
            ir_frame = frames.get_infrared_frame()

            if not depth_frame or not color_frame or not ir_frame:
                print("⚠️ One or more frames not available.")
                continue

            # Convert images to numpy arrays
            depth_image = np.asanyarray(depth_frame.get_data())
            color_image = np.asanyarray(color_frame.get_data())
            ir_image = np.asanyarray(ir_frame.get_data())

            # Normalize depth for display
            depth_colormap = cv2.convertScaleAbs(depth_image, alpha=0.03)

            # Display images
            cv2.imshow('Depth', depth_colormap)
            cv2.imshow('Color', color_image)
            cv2.imshow('Infrared', ir_image)

            # Break on ESC
            if cv2.waitKey(1) == 27:
                print("🛑 ESC pressed. Exiting...")
                break

    except Exception as e:
        print("❌ Error initializing streams:", str(e))

    finally:
        pipeline.stop()
        cv2.destroyAllWindows()
        print("🧹 Cleaned up. Goodbye.")

if __name__ == "__main__":
    main()

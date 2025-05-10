import pyrealsense2 as rs

# 1. Start the RealSense pipeline
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)  # resolution must match your usage
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)

pipeline_profile = pipeline.start(config)

# 2. Get the intrinsics
color_stream = pipeline_profile.get_stream(rs.stream.color)
intr = color_stream.as_video_stream_profile().get_intrinsics()

fx = intr.fx
fy = intr.fy
cx = intr.ppx
cy = intr.ppy

print("fx =", fx, "fy =", fy, "cx =", cx, "cy =", cy)

# 3. Stop the pipeline
pipeline.stop()

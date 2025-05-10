import pyrealsense2 as rs

# Start the pipeline
pipeline = rs.pipeline()
config   = rs.config()
profile  = pipeline.start(config)

# Get the depth sensor
depth_sensor = profile.get_device().first_depth_sensor()

preset_range = depth_sensor.get_option_range(rs.option.visual_preset)
print("Available presets:")
for i in range(int(preset_range.max) + 1):
    name = depth_sensor.get_option_value_description(rs.option.visual_preset, i)
    print(f"Preset {i}: {name}")

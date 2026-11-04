#!/usr/bin/env python3
"""Measure the head camera's auto exposure and save the fixed exposure for this run.

The start hook runs this before Dora starts:

    lock_exposure.py <camera_head.yaml> <output file> --scale <factor>

nodes/run_camera.sh then gives the saved value to both cameras, so the image does not change while
the skill runs, yet follows the lighting of the day. The SDK writes a Log/ folder into the current
directory.

Auto exposure reads gain 256 at its base gain (booth lighting, 2026-11-04). The cameras run at the
fixed gain in their configs, so any extra auto gain is converted into exposure time. That
conversion is untested: only gain 256 has been seen so far.
"""

import argparse
import sys
import time
from pathlib import Path

import pyorbbecsdk as ob
import yaml

SETTLE_S = 3.0  # auto exposure settled within about 1 s in every measurement
BASE_AUTO_GAIN = 256


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("config", type=Path, help="camera_head.yaml")
    parser.add_argument("output", type=Path, help="file receiving the fixed exposure in us")
    parser.add_argument("--scale", type=float, required=True, help="fixed exposure / auto exposure")
    args = parser.parse_args()

    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    serial = config["device_serial"]
    stream = config["color"]["data_flow"]

    context = ob.Context()
    devices = context.query_devices()
    if devices.get_count() == 0:
        sys.exit("no Orbbec device found (USB connection, udev rule)")
    device = devices.get_device_by_serial_number(serial)
    pipeline = ob.Pipeline(device)
    profiles = pipeline.get_stream_profile_list(ob.OBSensorType.COLOR_SENSOR)
    streams = ob.Config()
    streams.enable_stream(
        profiles.get_video_stream_profile(stream["width"], stream["height"], ob.OBFormat.RGB, stream["fps"])
    )

    props = ob.OBPropertyID
    pipeline.start(streams)
    # The device keeps the manual setting of the previous run.
    device.set_bool_property(props.OB_PROP_COLOR_AUTO_EXPOSURE_BOOL, True)
    time.sleep(SETTLE_S)
    if not pipeline.wait_for_frames(2000):
        sys.exit(f"head camera {serial} produced no frames")
    exposure = device.get_int_property(props.OB_PROP_COLOR_EXPOSURE_INT)
    gain = device.get_int_property(props.OB_PROP_COLOR_GAIN_INT)
    limit = device.get_int_property_range(props.OB_PROP_COLOR_EXPOSURE_INT).max
    pipeline.stop()

    fixed = min(round(exposure * gain / BASE_AUTO_GAIN * args.scale), limit)
    if fixed <= 0:
        sys.exit(f"head camera {serial} reported auto exposure {exposure} us, gain {gain}")
    args.output.write_text(f"{fixed}\n", encoding="utf-8")
    print(f"exposure: head auto {exposure} us, gain {gain}, scale {args.scale} -> fixed {fixed} us")


if __name__ == "__main__":
    main()

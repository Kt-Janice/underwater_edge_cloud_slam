#!/usr/bin/env python3

import argparse
import os
import sys
import time

import cv2
import rosbag
import rospy

from cv_bridge import CvBridge
from sensor_msgs.msg import Image


DEFAULT_TOPIC = "/davis_left/image_raw"
DEFAULT_FRAME_ID = "davis_left"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Convert MP4 video to ROS1 rosbag."
    )

    parser.add_argument(
        "--input",
        "-i",
        required=True,
        help="Input MP4 video path."
    )

    parser.add_argument(
        "--output",
        "-o",
        required=True,
        help="Output rosbag path."
    )

    parser.add_argument(
        "--encoding",
        choices=["bgr8", "mono8"],
        default="mono8",
        help="ROS image encoding. Default: mono8."
    )

    parser.add_argument(
        "--fps",
        type=float,
        default=None,
        help="Override video FPS. Default: read from MP4 metadata."
    )

    parser.add_argument(
        "--topic",
        default=DEFAULT_TOPIC,
        help=f"ROS image topic. Default: {DEFAULT_TOPIC}"
    )

    parser.add_argument(
        "--frame-id",
        default=DEFAULT_FRAME_ID,
        help=f"Image frame_id. Default: {DEFAULT_FRAME_ID}"
    )

    parser.add_argument(
        "--compression",
        choices=["none", "lz4", "bz2"],
        default="lz4",
        help="Rosbag compression. Default: lz4."
    )

    return parser.parse_args()


def get_compression(name):
    if name == "lz4":
        return rosbag.Compression.LZ4

    if name == "bz2":
        return rosbag.Compression.BZ2

    return rosbag.Compression.NONE


def main():
    args = parse_args()

    if not os.path.isfile(args.input):
        print(f"[ERROR] Input video does not exist: {args.input}")
        return 1

    output_dir = os.path.dirname(os.path.abspath(args.output))

    if not os.path.isdir(output_dir):
        print(f"[ERROR] Output directory does not exist: {output_dir}")
        return 1

    cap = cv2.VideoCapture(args.input)

    if not cap.isOpened():
        print(f"[ERROR] Failed to open video: {args.input}")
        return 1

    video_fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    metadata_frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if args.fps is not None:
        fps = args.fps
    else:
        fps = video_fps

    if fps is None or fps <= 0.0:
        print(
            "[ERROR] Invalid FPS. "
            "Please specify FPS manually using --fps."
        )
        cap.release()
        return 1

    frame_interval = 1.0 / fps

    # 获取一次真实 Unix 时间，作为视频第一帧时间戳
    base_timestamp = time.time()

    bridge = CvBridge()
    compression = get_compression(args.compression)

    print("============================================================")
    print("MP4 -> ROS1 rosbag")
    print("============================================================")
    print(f"Input             : {args.input}")
    print(f"Output            : {args.output}")
    print(f"Resolution        : {width} x {height}")
    print(f"Video FPS         : {video_fps:.9f}")
    print(f"Used FPS          : {fps:.9f}")
    print(f"Metadata frames   : {metadata_frame_count}")
    print(f"Topic             : {args.topic}")
    print(f"Frame ID          : {args.frame_id}")
    print(f"Encoding          : {args.encoding}")
    print(f"Compression       : {args.compression}")
    print(f"Base timestamp    : {base_timestamp:.9f}")
    print("============================================================")

    frame_index = 0

    try:
        with rosbag.Bag(
            args.output,
            "w",
            compression=compression
        ) as bag:

            while True:
                ret, frame = cap.read()

                if not ret:
                    break

                if frame is None:
                    print(
                        f"\n[ERROR] Invalid frame at index "
                        f"{frame_index}"
                    )
                    break

                if args.encoding == "mono8":
                    if len(frame.shape) == 3:
                        frame = cv2.cvtColor(
                            frame,
                            cv2.COLOR_BGR2GRAY
                        )

                timestamp_sec = (
                    base_timestamp
                    + frame_index * frame_interval
                )

                stamp = rospy.Time.from_sec(timestamp_sec)

                image_msg = bridge.cv2_to_imgmsg(
                    frame,
                    encoding=args.encoding
                )

                image_msg.header.seq = frame_index
                image_msg.header.stamp = stamp
                image_msg.header.frame_id = args.frame_id

                bag.write(
                    args.topic,
                    image_msg,
                    t=stamp
                )

                frame_index += 1

                if frame_index % 100 == 0:
                    print(
                        f"\rProcessed: {frame_index} frames",
                        end="",
                        flush=True
                    )

    except Exception as e:
        cap.release()

        print()
        print(f"[ERROR] Conversion failed: {e}")

        return 1

    cap.release()

    if frame_index == 0:
        print()
        print("[ERROR] No frames written.")
        return 1

    duration = (frame_index - 1) / fps
    last_timestamp = base_timestamp + duration

    print()
    print("============================================================")
    print("Conversion completed")
    print("============================================================")
    print(f"Frames          : {frame_index}")
    print(f"Duration        : {duration:.6f} s")
    print(f"FPS             : {fps:.9f}")
    print(f"First timestamp : {base_timestamp:.9f}")
    print(f"Last timestamp  : {last_timestamp:.9f}")
    print(f"Topic           : {args.topic}")
    print(f"Output          : {args.output}")
    print("============================================================")

    return 0


if __name__ == "__main__":
    sys.exit(main())

#!/usr/bin/env python3

"""Publish one raw bag image repeatedly with advancing simulated time.

This is a LAND/AIR NOT_INITIALIZED regression harness.  It deliberately keeps
the pixels fixed, while publishing an increasing /clock and Image header stamp
so the frontend receives normal, monotonically-timestamped input for a
controlled duration.
"""

import argparse
import copy
import sys
import time

import rosbag
import rospy
from rosgraph_msgs.msg import Clock
from sensor_msgs.msg import Image


def read_first_image(bag_path, topic):
    with rosbag.Bag(bag_path, "r") as bag:
        for _, message, _ in bag.read_messages(topics=[topic]):
            # rosbag can deserialize through a Python class object distinct
            # from this process's sensor_msgs.Image class; ROS type metadata
            # is the stable compatibility check here.
            if getattr(message, "_type", "") == "sensor_msgs/Image":
                return copy.deepcopy(message)
    raise RuntimeError("no sensor_msgs/Image message found on {}".format(topic))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--bag", required=True)
    parser.add_argument("--input-topic", required=True)
    parser.add_argument("--output-topic", required=True)
    parser.add_argument("--duration", type=float, default=65.0)
    parser.add_argument("--rate", type=float, default=20.0)
    args = parser.parse_args()

    if args.duration <= 0.0:
        raise ValueError("duration must be positive")
    if args.rate <= 0.0:
        raise ValueError("rate must be positive")

    image = read_first_image(args.bag, args.input_topic)
    rospy.init_node("land_air_static_image_regression_publisher")
    image_publisher = rospy.Publisher(args.output_topic, Image, queue_size=1)
    clock_publisher = rospy.Publisher("/clock", Clock, queue_size=1)

    period = 1.0 / args.rate
    initial_stamp = image.header.stamp.to_sec()
    if initial_stamp <= 0.0:
        initial_stamp = 1.0

    start = time.monotonic()
    next_deadline = start
    count = 0
    while not rospy.is_shutdown():
        elapsed = time.monotonic() - start
        if elapsed >= args.duration:
            break

        stamp = rospy.Time.from_sec(initial_stamp + count * period)
        clock_publisher.publish(Clock(clock=stamp))
        image.header.stamp = stamp
        image_publisher.publish(image)
        count += 1

        next_deadline += period
        remaining = next_deadline - time.monotonic()
        if remaining > 0.0:
            time.sleep(remaining)

    sys.stdout.write("published {} static frames\\n".format(count))


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Camera Performance Test Script

Monitors ZED camera topics, tracks frame counts and dropped frames,
and writes periodic summaries to CSV.
"""

import argparse
import csv
import signal
import sys
import threading
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import rclpy
import yaml
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from rclpy.time import Time
from sensor_msgs.msg import Image

# Global state for tracking
frame_counts: Dict[str, int] = defaultdict(int)
dropped_by_sequence: Dict[str, int] = defaultdict(int)
dropped_by_timestamp: Dict[str, int] = defaultdict(int)
last_sequence: Dict[str, Optional[int]] = defaultdict(lambda: None)
last_timestamp: Dict[str, Optional[Time]] = defaultdict(lambda: None)

# CSV file handle
csv_file = None
csv_writer = None
shutdown_requested = False


def load_yaml_config(config_path: str) -> List[str]:
    """
    Load topic list from YAML configuration file.

    Args:
        config_path: Path to YAML file

    Returns:
        List of topic names

    Raises:
        FileNotFoundError: If config file doesn't exist
        ValueError: If config format is invalid
    """
    config_file = Path(config_path)
    if not config_file.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    try:
        with open(config_file, "r") as f:
            config = yaml.safe_load(f)

        if not isinstance(config, dict):
            raise ValueError("YAML config must be a dictionary")

        if "topics" not in config:
            raise ValueError("YAML config must contain 'topics' key")

        topics = config["topics"]
        if not isinstance(topics, list):
            raise ValueError("'topics' must be a list")

        # Validate topic names
        validated_topics = []
        for topic in topics:
            if not isinstance(topic, str) or not topic.strip():
                raise ValueError(f"Invalid topic name: {topic}")
            validated_topics.append(topic.strip())

        if not validated_topics:
            raise ValueError("Topics list cannot be empty")

        return validated_topics

    except yaml.YAMLError as e:
        raise ValueError(f"Invalid YAML format: {e}")
    except Exception as e:
        raise ValueError(f"Error loading config: {e}")


def initialize_csv(output_path: str):
    """Initialize CSV file and writer."""
    global csv_file, csv_writer
    try:
        csv_file = open(output_path, "w", newline="")
        fieldnames = [
            "timestamp",
            "topic",
            "frames_captured",
            "dropped_by_sequence",
            "dropped_by_timestamp",
            "total_dropped",
        ]
        csv_writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        csv_writer.writeheader()
        csv_file.flush()
        print(f"CSV file initialized: {output_path}")
    except Exception as e:
        print(f"Failed to initialize CSV file: {e}", file=sys.stderr)
        raise


def message_callback(msg: Image, topic: str, expected_frame_duration: float):
    """Callback for received messages."""
    try:
        # Update frame count
        frame_counts[topic] += 1

        # Extract timestamp
        msg_time = Time.from_msg(msg.header.stamp)

        # Check for dropped frames by sequence number
        if hasattr(msg.header, "seq") and msg.header.seq is not None:
            try:
                current_seq = msg.header.seq
                if last_sequence[topic] is not None:
                    seq_diff = current_seq - last_sequence[topic]
                    if seq_diff > 1:
                        dropped = seq_diff - 1
                        dropped_by_sequence[topic] += dropped
                last_sequence[topic] = current_seq
            except Exception:
                pass

        # Check for dropped frames by timestamp
        if last_timestamp[topic] is not None:
            try:
                time_diff = (msg_time - last_timestamp[topic]).nanoseconds / 1e9
                # Detect drops if time gap is > 1.5x expected interval
                if time_diff > 1.5 * expected_frame_duration:
                    estimated_dropped = int((time_diff / expected_frame_duration) - 1)
                    if estimated_dropped > 0:
                        dropped_by_timestamp[topic] += estimated_dropped
            except Exception:
                pass

        # Update last timestamp
        last_timestamp[topic] = msg_time

    except Exception as e:
        print(f"Error in message callback for {topic}: {e}", file=sys.stderr)


def write_csv_summary(topics: List[str]):
    """Write periodic summary to CSV."""
    global csv_file, csv_writer
    if csv_file is None or csv_writer is None:
        return

    try:
        current_time = datetime.now().isoformat()
        for topic in topics:
            total_dropped = (
                dropped_by_sequence[topic] + dropped_by_timestamp[topic]
            )
            row = {
                "timestamp": current_time,
                "topic": topic,
                "frames_captured": frame_counts[topic],
                "dropped_by_sequence": dropped_by_sequence[topic],
                "dropped_by_timestamp": dropped_by_timestamp[topic],
                "total_dropped": total_dropped,
            }
            csv_writer.writerow(row)
        csv_file.flush()
    except Exception as e:
        print(f"Error writing CSV summary: {e}", file=sys.stderr)


def csv_writer_thread(topics: List[str], interval: float):
    """Background thread that writes CSV summaries periodically."""
    while not shutdown_requested:
        write_csv_summary(topics)
        time.sleep(interval)


def signal_handler(signum, frame):
    """Handle shutdown signals gracefully."""
    global shutdown_requested
    print(f"\nShutdown signal {signum} received. Writing final summary...")
    shutdown_requested = True
    if csv_file:
        write_csv_summary(list(frame_counts.keys()))
        try:
            csv_file.close()
        except Exception:
            pass
    rclpy.shutdown()
    sys.exit(0)


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Monitor camera topics and track frame performance"
    )
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to YAML config file with topics list",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Path to output CSV file (default: camera_performance_<timestamp>.csv)",
    )
    parser.add_argument(
        "--summary-interval",
        type=float,
        default=1.0,
        help="Interval in seconds for writing CSV summaries (default: 1.0)",
    )
    parser.add_argument(
        "--expected-fps",
        type=float,
        default=30.0,
        help="Expected frame rate for drop detection (default: 30.0)",
    )

    args = parser.parse_args()

    # Load configuration
    try:
        topics = load_yaml_config(args.config)
        print(f"Loaded {len(topics)} topics from config:")
        for topic in topics:
            print(f"  - {topic}")
    except Exception as e:
        print(f"Error loading config: {e}", file=sys.stderr)
        sys.exit(1)

    # Determine output CSV path
    if args.output:
        output_csv = args.output
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_csv = f"camera_performance_{timestamp}.csv"

    # Initialize CSV
    try:
        initialize_csv(output_csv)
    except Exception as e:
        print(f"Failed to initialize CSV: {e}", file=sys.stderr)
        sys.exit(1)

    # Setup signal handlers
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    # Initialize ROS 2
    rclpy.init()

    # Create a minimal node for subscriptions
    node = Node("camera_performance_monitor")
    expected_frame_duration = 1.0 / args.expected_fps

    # Create subscriptions
    subscriptions = []
    for topic in topics:
        try:
            callback = lambda msg, t=topic: message_callback(
                msg, t, expected_frame_duration
            )
            sub = node.create_subscription(
                Image, topic, callback, qos_profile_sensor_data
            )
            subscriptions.append(sub)
            print(f"Subscribed to topic: {topic}")
        except Exception as e:
            print(f"Failed to subscribe to {topic}: {e}. Continuing...", file=sys.stderr)

    if not subscriptions:
        print("No subscriptions created. Exiting.", file=sys.stderr)
        node.destroy_node()
        rclpy.shutdown()
        sys.exit(1)

    # Start CSV writer thread
    csv_thread = threading.Thread(
        target=csv_writer_thread, args=(topics, args.summary_interval), daemon=True
    )
    csv_thread.start()

    print(f"\nCamera Performance Monitor started.")
    print(f"Monitoring {len(topics)} topics")
    print(f"Output CSV: {output_csv}")
    print(f"Summary interval: {args.summary_interval}s")
    print(f"Expected FPS: {args.expected_fps}")
    print("Press Ctrl+C to stop.\n")

    # Spin the node
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
    finally:
        shutdown_requested = True
        write_csv_summary(topics)
        if csv_file:
            csv_file.close()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()

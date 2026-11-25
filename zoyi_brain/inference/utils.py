import itertools

import cv2
import numpy as np
from cv_bridge import CvBridge
from message_filters import ApproximateTimeSynchronizer
from rclpy.duration import Duration
from rclpy.time import Time

bridge = CvBridge()


def pack_z02_state(positions: np.ndarray, joint_dim: int, gripper_threshold: float) -> np.ndarray:
    """Pack joint feedback and leave the 25-D base velocity slots at zero."""
    state = np.zeros(joint_dim, dtype=np.float32)
    state[:7] = positions[4:11]
    state[7] = np.linalg.norm(positions[[22, 24, 25, 27]]) <= gripper_threshold
    state[8:15] = positions[11:18]
    state[15] = np.linalg.norm(positions[[32, 34, 35, 37]]) <= gripper_threshold
    if joint_dim == 25:
        state[16:20] = positions[:4]
        state[20:22] = positions[18:20]
    return state


def stamp_to_sec(stamp) -> float:
    if hasattr(stamp, "nanoseconds"):
        return stamp.nanoseconds * 1e-9
    return float(stamp.sec) + float(stamp.nanosec) * 1e-9


def compute_timestamp_offsets(stamps_sec, ref_idx: int = 0):
    """Per-topic offsets relative to stamps_sec[ref_idx]."""
    ref = stamps_sec[ref_idx]
    return [s - ref for s in stamps_sec]


def align_timestamp(stamp_sec: float, offset_sec: float) -> float:
    """Map a raw stamp into the reference topic's time base.

    Does not mutate any message header.stamp.
    """
    return stamp_sec - offset_sec


class TimestampAlignedSynchronizer(ApproximateTimeSynchronizer):
    """Approximate sync using offset-aligned stamps; never mutates headers.

    Offsets are estimated once from the first stamp seen on each topic, then
    matching uses align_timestamp(...) with the configured slop.
    """

    def __init__(self, fs, queue_size, slop, ref_index: int = 0):
        super().__init__(fs, queue_size, slop)
        self.ref_index = ref_index
        self.offsets = None
        self._latest_raw_sec = [None] * len(fs)
        self._latest_msg = [None] * len(fs)

    def _enqueue_aligned(self, msg, my_queue, my_queue_index):
        raw_sec = stamp_to_sec(msg.header.stamp)
        aligned_sec = align_timestamp(raw_sec, self.offsets[my_queue_index])
        stamp = Time(nanoseconds=int(round(aligned_sec * 1e9)))
        my_queue[stamp.nanoseconds] = msg
        while len(my_queue) > self.queue_size:
            del my_queue[min(my_queue)]
        return stamp

    def add(self, msg, my_queue, my_queue_index=None):
        raw_sec = stamp_to_sec(msg.header.stamp)
        self.lock.acquire()
        try:
            if my_queue_index is not None:
                self._latest_raw_sec[my_queue_index] = raw_sec
                self._latest_msg[my_queue_index] = msg
            if self.offsets is None:
                if any(s is None for s in self._latest_raw_sec):
                    return
                self.offsets = compute_timestamp_offsets(
                    list(self._latest_raw_sec), ref_idx=self.ref_index
                )
                # Seed all queues once offsets are known.
                for i, latest in enumerate(self._latest_msg):
                    self._enqueue_aligned(latest, self.queues[i], i)

            stamp = self._enqueue_aligned(msg, my_queue, my_queue_index)

            if my_queue_index is None:
                search_queues = self.queues
            else:
                search_queues = (
                    self.queues[:my_queue_index]
                    + self.queues[my_queue_index + 1 :]
                )

            stamps = []
            for queue in search_queues:
                topic_stamps = []
                for s in queue:
                    stamp_delta = Duration(
                        nanoseconds=abs(s - stamp.nanoseconds)
                    )
                    if stamp_delta > self.slop:
                        continue
                    topic_stamps.append(
                        (
                            Time(
                                nanoseconds=s,
                                clock_type=stamp.clock_type,
                            ),
                            stamp_delta,
                        )
                    )
                if not topic_stamps:
                    return
                topic_stamps = sorted(topic_stamps, key=lambda x: x[1])
                stamps.append(topic_stamps)

            for vv in itertools.product(
                *[list(zip(*s))[0] for s in stamps]
            ):
                vv = list(vv)
                if my_queue_index is not None:
                    vv.insert(my_queue_index, stamp)
                qt = list(zip(self.queues, vv))
                if ((max(vv) - min(vv)) < self.slop) and (
                    len([1 for q, t in qt if t.nanoseconds not in q]) == 0
                ):
                    msgs = [q[t.nanoseconds] for q, t in qt]
                    self.signalMessage(*msgs)
                    for q, t in qt:
                        del q[t.nanoseconds]
                    break
        finally:
            self.lock.release()


def process_bgr_rgb(img_bgr, rotate=False, roi=None, img_size=(640, 360)):
    """Convert a frame hub image to HWC RGB at the requested resolution."""
    if roi is not None:
        img_bgr = img_bgr[roi[1]:roi[3], roi[0]:roi[2], :]
    if rotate:
        img_bgr = cv2.rotate(img_bgr, cv2.ROTATE_180)

    h, w = img_bgr.shape[:2]
    target_w, target_h = img_size
    if (w, h) != (target_w, target_h):
        img_bgr = cv2.resize(img_bgr, img_size, interpolation=cv2.INTER_AREA)

    return cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)


def process_bgr_array(
    img_bgr, rotate=False, roi=None, img_size=(640, 360), as_numpy=False
):
    """Convert a BGR uint8 image (e.g. from frame hub) to model CHW RGB float."""
    img_rgb = process_bgr_rgb(img_bgr, rotate=rotate, roi=roi, img_size=img_size)
    img = np.transpose(img_rgb.astype(np.float32), (2, 0, 1)) / 255.0
    if as_numpy:
        return img

    import torch

    return torch.from_numpy(img)


def process_img_msg(
    img_msg, rotate=False, roi=None, img_size=(640, 360), as_numpy=False
):
    # Convert ROS msg to OpenCV image (BGR)
    img_bgr = bridge.imgmsg_to_cv2(img_msg, "bgr8")
    return process_bgr_array(
        img_bgr, rotate=rotate, roi=roi, img_size=img_size, as_numpy=as_numpy)


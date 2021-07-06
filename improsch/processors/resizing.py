from typing import Iterable, List, Tuple
import cv2


def resize(img, dst_size: Tuple[int, int]):
    return cv2.resize(img, dst_size)


def resize_batch(images: Iterable, dst_size: Tuple[int, int]) -> Iterable:
    return [resize(img, dst_size) for img in images]

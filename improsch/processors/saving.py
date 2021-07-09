from typing import Iterable, List
import numpy as np
import cv2
import logging
from multiprocessing import Pool
from math import ceil
from pathlib import Path
import os


def save_image(img: np.ndarray, filepath: str):
    cv2.imwrite(filepath, img)


def save_all(images: Iterable[np.ndarray], filepaths: Iterable[str]):
    for img, filepath in zip(images, filepaths):
        try:
            save_image(img, filepath)
            logging.info("Saved image to", filepath)
        except Exception as err:
            logging.error(f"Error while saving: {str(err)}")


def unpack_func_for_saving(argument):
    images, filepaths = argument
    return save_all(images, filepaths)


def save_multiprocess(images: List[np.ndarray], filepaths: List[str], pool_size: int, storage_path: str):
    assert len(images) == len(filepaths), RuntimeError("Lengths of images and filepaths are different")

    logging.info("Saving", len(images), "into storage", storage_path)

    new_filepaths = []
    for i, filename in enumerate(filepaths):
        new_filepaths.append(os.path.join(storage_path, Path(filename).name))

    chunk_size = ceil(len(images) / pool_size)
    chunks = []

    for i in range(0, len(images), chunk_size):
        chunks.append((
            images[i:i+chunk_size],
            new_filepaths[i:i+chunk_size]
        ))

    pool = Pool(pool_size)
    pool.map(unpack_func_for_saving, chunks)
    pool.close()
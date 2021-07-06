from typing import Callable, List, Optional
import cv2
import numpy as np
import os


def get_reader(filter_func: Optional[Callable[[np.ndarray], bool]]):
    def read(data_dir: str) -> np.ndarray:
        images: List[np.ndarray] = []
        imagenames = []

        for root, subdirs, files in os.walk(data_dir):
            filenames = map(lambda x: os.path.join(root, x), files)

            for filename in filenames:
                images = []

                for name in filenames:
                    try:
                        img = cv2.imread(name)
                        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

                        if not filter_func or filter_func(img):
                            images.append(img)
                            imagenames.append(filename)
                    except Exception:
                        # TODO alerting
                        continue

            return images

        return read
"""Module to handle deduplication of images.
Includes model for features extraction and model for neighbours searching."""
import os
from typing import Iterable, List, Optional, Sized, Tuple, Union
import torch
import faiss
import numpy as np
import cv2
from multiprocessing import Pool
from torch.multiprocessing import Pool as torch_pool
from functools import partial
import json


CONFIG_NAME = 'config.json'


class FeatureExtractor:
    pass


class NNfeatureExtractor(FeatureExtractor, torch.nn.Module):
    def __init__(self, torch_model_name: str, repo: str = 'pytorch/vision:v0.9.0'):
        if torch_model_name != 'googlenet':
            raise AttributeError("Not supported model:", torch_model_name)

        torch.nn.Module.__init__(self)

        self.model = torch.hub.load(repo, torch_model_name, pretrained=True)
        self.model.eval()
        self.model.fc = torch.nn.Identity()

    def get_embeddings(
        self,
        images: Union[torch.Tensor, np.ndarray],
        batch_size: int,
        device: str = 'cuda'
    ) -> np.ndarray:
        if not isinstance(images, torch.Tensor):
            images = torch.Tensor(images).permute(0, 3, 1, 2)

        dataset = torch.utils.data.TensorDataset(images)
        dataloader = torch.utils.data.DataLoader(
            dataset, shuffle=False, drop_last=False,
            batch_size=batch_size  # TODO num_workers and pin_memory
        )

        self.model.to(device)
        embeddings = []

        with torch.no_grad():
            for image_batch, in dataloader:
                image_batch = image_batch.to(device)
                embed = self.model(image_batch).cpu().numpy()
                embeddings.append(embed)

        return np.concatenate(embeddings)


class ImageIndex:
    def __init__(self, index_dir: Optional[str] = None,  feat_dim: Optional[int] = None):
        if not (index_dir or feat_dim is not None):
            raise AttributeError()

        if index_dir:
            self.index = faiss.read_index(os.path.join(index_dir, 'index.faiss'))
            with open(os.path.join(index_dir, CONFIG_NAME)) as file:
                self.index_config = json.load(file)
        else:
            self.index = faiss.index_factory(feat_dim, "Flat", faiss.METRIC_INNER_PRODUCT)
            self.index_config = {
                'index_length': 0,
                'feat_dim': feat_dim,
                'index_type': 'Flat',
                'metric': 'inner_product'
            }

    def add_vectors(self, vectors):
        # we found cosine similarity using inner product
        # so vectors shoul be normalized
        vectors = np.array(vectors)
        faiss.normalize_L2(vectors)
        self.index.add(vectors)
        self.index_config['index_length'] += len(vectors)

    def find_neighbours(self, vectors):
        # we found cosine similarity using inner product
        # so vectors shoul be normalized
        vectors = np.array(vectors)
        faiss.normalize_L2(vectors)
        distances, indexes = self.index.search(vectors, 2)

        neighbours = []
        for i in range(distances.shape[0]):
            neighbours.append((i, indexes[i, 1], distances[i, 1]))

        # maximum values on top
        neighbours.sort(key=lambda x: 1 - x[-1])

        return neighbours

    def get_index_length(self):
        return self.index_config['index_length']

    def save_on_disk(self, directory_path: str):
        faiss.write_index(self.index, os.path.join(directory_path, 'index.faiss'))
        with open(os.path.join(directory_path, CONFIG_NAME), 'w') as file:
            json.dump(self.index_config, file)


class Deduplicator:
    def __init__(self, pool_size: int) -> None:
        self.feature_extractor = NNfeatureExtractor('googlenet')
        self.index = ImageIndex(feat_dim=1024)
        self.pool_size = pool_size

    def _get_filepaths(self, data_dir, chunk_size):
        all_filepaths = []
        for root, subdirs, files in os.walk(data_dir):
            for i, filename in enumerate(files):
                if i % chunk_size == 0:
                    all_filepaths.append([])
                all_filepaths[-1].append(os.path.join(root, filename))

        return all_filepaths

    def read_images(self, filepaths, dst_size):
        '''Read group of images, resize and return along with the group mean and std'''
        images = []
        imagenames = []

        img_mean, img_std = 0, 0
        count_images = 0

        for filename in filepaths:
            img = cv2.imread(filename)

            if img is None:
                # probably this file isn't an image
                print("Loading error:", filename[32:])
                continue

            if img.shape[0] < dst_size[0] or img.shape[1] < dst_size[1]:
                print(filename[32:], "size is", img.shape, "less than", dst_size)

            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, dst_size) / 255.
            images.append(img)
            imagenames.append(filename)

            # calculates for every channel
            img_mean += np.mean(img, axis=tuple(range(img.ndim-1)))
            img_std += np.std(img, axis=tuple(range(img.ndim-1)))
            count_images += 1

        if not len(images):
            return [], [], 0, 0

        images = np.stack(images)

        img_mean /= count_images
        img_std /= count_images

        return images, imagenames, img_mean, img_std

    def _get_embeddings(self, images, batch_size: int):
        embeddings = self.feature_extractor.get_embeddings(images, batch_size)
        return embeddings

    def _get_embeddings_from_dir(
        self,
        filepaths: List,
        dst_size: Union[Tuple[int, int], List[int]],
        batch_size: int
    ):
        cpu_pool = Pool(min(self.pool_size, len(filepaths)))
        read_func = partial(self.read_images, dst_size=dst_size)
        mp_result: List[Tuple] = cpu_pool.map(read_func, filepaths)
        cpu_pool.terminate()

        images = []
        imagenames = []
        img_mean, img_std = 0., 0.
        count_chunks = 0

        for image_chunk, names_chunk, mean_, std_, in mp_result:
            if len(image_chunk):
                images.extend(image_chunk)
                imagenames.extend(names_chunk)

                img_mean += mean_
                img_std += std_
                count_chunks += 1

        img_mean /= count_chunks
        img_std /= count_chunks
        images = np.stack(images)

        embeddings = self._get_embeddings(images, batch_size)

        return embeddings, imagenames

    def process_embeddings(self, embeddings, imagenames: List[str], data_dir: str):
        index_length = self.index.get_index_length()
        indexes_path = os.path.join(data_dir, 'file_indexes.json')
        if os.path.isfile(indexes_path):
            with open(indexes_path) as file:
                ids_to_filenames = json.load(file)
        else:
            ids_to_filenames = {}

        for i, imagename in zip(
            range(index_length, index_length + len(imagenames)),
            imagenames
        ):
            ids_to_filenames[i] = imagename

        with open(os.path.join(data_dir, 'file_indexes.json'), 'w') as file:
            json.dump(ids_to_filenames, file)

        self.index.add_vectors(embeddings)

        neighbours = self.index.find_neighbours(embeddings)
        neighbours = [
            (ids_to_filenames[ind1], ids_to_filenames[ind2], float(distance))
            for ind1, ind2, distance in neighbours
        ]

        return neighbours

    def run_deduplication(
        self,
        data_dir: str,
        dst_size: Union[Tuple[int, int], List[int]],
        chunk_size: Optional[int] = None
    ):
        filepaths = self._get_filepaths(data_dir, chunk_size)
        # TODO batch_size as parameter
        embeddings, imagenames = self._get_embeddings_from_dir(filepaths, dst_size, 64)

        neighbours = self.process_embeddings(embeddings, imagenames, data_dir)
        return neighbours

    def __call__(self, images: List[np.ndarray], imagenames: List[str], data_dir: str, batch_size: int) -> None:
        embeddings = self._get_embeddings(images, batch_size)
        neighbours = self.process_embeddings(embeddings, imagenames, data_dir)
        # TODO save index in parallel thread
        return neighbours

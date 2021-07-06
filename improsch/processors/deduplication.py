"""Module to handle deduplication of images.
Includes model for features extraction and model for neighbours searching."""
from typing import Optional, Union
import torch
import faiss
import numpy as np


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
            images = torch.Tensor(images)

        dataset = torch.utils.data.TensorDataset(images)
        dataloader = torch.utils.data.DataLoader(
            dataset, shuffle=False, drop_last=False,
            batch_size=batch_size  # TODO num_workers and pin_memory
        )

        self.model.to(device)
        embeddings = []

        with torch.no_grad():
            for image_batch in dataloader:
                image_batch = image_batch.to(device).cpu().numpy()
                embed = self.model(image_batch)
                embeddings.append(embed)

        return np.stack(embeddings)


class ImageIndex:
    def __init__(self, index_path: Optional[str] = None,  feat_dim: Optional[int] = None):
        if not (index_path or feat_dim is not None):
            raise AttributeError()

        if index_path:
            self.index = faiss.read_index(index_path)
        else:
            self.index = faiss.index_factory(feat_dim, "Flat", faiss.METRIC_INNER_PRODUCT)

    def add_vectors(self, vectors):
        # we found cosine similarity using inner product
        # so vectors shoul be normalized
        faiss.normalize_L2(np.array(vectors))
        self.index.add(vectors)

    def find_neighbours(self, vectors):
        # we found cosine similarity using inner product
        # so vectors shoul be normalized
        faiss.normalize_L2(np.array(vectors))
        distances, indexes = self.index.search(vectors, 2)

        neighbours = []
        for i in range(distances.shape[0]):
            neighbours.append((i, indexes[i, 1], distances[i, 1]))

        # maximum values on top
        neighbours.sort(key=lambda x: 1 - x[-1])

        return neighbours

    def save_on_disk(self, path: str):
        faiss.write_index(self.index, path)

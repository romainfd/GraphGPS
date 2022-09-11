from typing import Optional, Callable, List

import os
import glob
import os.path as osp

import torch
from torch_geometric.data import (InMemoryDataset, Data, download_url,
                                  extract_tar, extract_zip)
from torch_geometric.utils import remove_isolated_nodes, from_networkx
from .GraphCoversRepo.covers import gen_graphCovers

"""
Implementing custom dataset into GraphGPS 
Ref.: 
- https://github.com/rampasek/GraphGPS/issues/3
- https://pytorch-geometric.readthedocs.io/en/latest/notes/create_dataset.html#creating-in-memory-datasets

This initial file is a local copy of MalNetTiny class from PyG
https://github.com/pyg-team/pytorch_geometric/blob/master/torch_geometric/datasets/malnet_tiny.py
then updated to fit into GraphGPS architecture
https://github.com/rampasek/GraphGPS/blob/main/graphgps/loader/dataset/malnet_tiny.py

It was then updated to plug into GraphCovers dataset in a compatible PyG and GraphGPS format
"""


class QM7(InMemoryDataset):
    r"""Loading QM7 dataset.

    Implemented as a custom PyG dataset
    Ref.: https://pytorch-geometric.readthedocs.io/en/latest/notes/create_dataset.html#creating-in-memory-datasets

    Args:
        root (string): Root directory where the dataset should be saved.
        transform (callable, optional): A function/transform that takes in an
            :obj:`torch_geometric.data.Data` object and returns a transformed
            version. The data object will be transformed before every access.
            (default: :obj:`None`)
        pre_transform (callable, optional): A function/transform that takes in
            an :obj:`torch_geometric.data.Data` object and returns a
            transformed version. The data object will be transformed before
            being saved to disk. (default: :obj:`None`)
        pre_filter (callable, optional): A function that takes in an
            :obj:`torch_geometric.data.Data` object and returns a boolean
            value, indicating whether the data object should be included in the
            final dataset. (default: :obj:`None`)
    """
    def __init__(self, root: str, transform: Optional[Callable] = None,
                 pre_transform: Optional[Callable] = None,
                 pre_filter: Optional[Callable] = None):
        super().__init__(root, transform, pre_transform, pre_filter)
        self.process()  # Forcing call as it was skipped when `processed_paths` files already existed
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self) -> List[str]:
        """A list of files in the `raw_dir` which needs to be found in order to skip the download."""
        return []

    @property
    def processed_file_names(self) -> List[str]:
        """A list of files in the `processed_dir` which needs to be found in order to skip the processing."""
        return ['data.pt', 'split_dict.pt']

    def download(self):
        """Downloads raw data into `raw_dir`."""
        pass

    def process(self):
        """Processes raw data and saves it into the `processed_dir`."""
        edge_index = [
            [1, 2], [2, 1],
            [2, 3], [3, 2],
            [2, 4], [4, 2],
            [3, 4], [4, 3],
            [4, 5], [5, 4],
            [5, 6], [6, 5],
            [5, 7], [7, 5],
            [6, 7], [7, 6]
        ]
        edge_index = list(map(lambda l: [i - 1 for i in l], edge_index))
        cycle_edge = [[1, 3], [4, 5]]

        graph_covers = gen_graphCovers(edge_index, degree=3, cycle_edge=cycle_edge, nb_covers=6)

        # make three classes:
        n = len(graph_covers)
        targets = torch.zeros(n, dtype=torch.long)
        targets[n // 3:] = 1
        targets[n // 3 * 2:] = 2

        # make 2 classes:
        # targets = torch.zeros(len(dgl_graphs), dtype=torch.long)
        # targets[len(dgl_graphs) // 2:] = 1

        # Simply duplicate train data for val and test
        data_list = []
        for _ in range(3):
            for i, cover in enumerate(graph_covers):
                graph = from_networkx(cover.nxGraph)
                graph.y = targets[i % n]
                data_list.append(graph)

        split_dict = {
            'train': list(range(n)),
            'valid': list(range(n, 2*n)),
            'test': list(range(2*n, 3*n))
        }

        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]

        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]

        torch.save(self.collate(data_list), self.processed_paths[0])
        torch.save(split_dict, self.processed_paths[1])

    def get_idx_split(self):
        return torch.load(self.processed_paths[1])

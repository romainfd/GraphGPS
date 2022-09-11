from typing import Optional, Callable, List
from scipy import io
from tqdm.auto import tqdm

import os
import glob
import os.path as osp

import numpy as np
import torch
from torch_geometric.data import (InMemoryDataset, Data, download_url,
                                  extract_tar, extract_zip)
from torch_geometric.utils import remove_isolated_nodes, from_networkx
from .GraphCoversRepo.covers import gen_graphCovers
import dgl

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
    # --------- TO UPDATE BASED ON NEED
    min_node, max_node = 4, 25
    embed_z = True
    use_positions = True
    splits = (.8, .1, .1)
    permutation_seed = 21

    url = 'http://deepchem.io.s3-website-us-west-1.amazonaws.com/datasets/qm7.mat'

    def __init__(self, root: str, transform: Optional[Callable] = None,
                 pre_transform: Optional[Callable] = None,
                 pre_filter: Optional[Callable] = None):
        super().__init__(root, transform, pre_transform, pre_filter)
        self.process()  # Forcing call as it was skipped when `processed_paths` files already existed
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self) -> List[str]:
        """A list of files in the `raw_dir` which needs to be found in order to skip the download."""
        return ['qm7.mat']

    @property
    def processed_file_names(self) -> List[str]:
        """A list of files in the `processed_dir` which needs to be found in order to skip the processing."""
        return ['data.pt', 'split_dict.pt']

    def download(self):
        """Downloads raw data into `raw_dir`."""
        download_url(self.url, self.raw_dir)

    def process(self):
        """Processes raw data and saves it into the `processed_dir`."""
        data = io.loadmat(osp.join(self.raw_dir, self.raw_file_names[0]))

        # ---------------- COPY PASTED FROM QGNN-FINAL
        # keys 'X', 'R', 'Z', 'T', 'P'
        labels = dgl.backend.tensor(data['T'], dtype=dgl.backend.data_type_dict['float32']).reshape(-1, 1)
        feats = data['X']
        num_graphs = labels.shape[0]
        graphs = []
        for i in range(num_graphs):
            edge_list = feats[i].nonzero()
            g = dgl.convert.graph(edge_list)
            g.edata['h'] = dgl.backend.tensor(feats[i][edge_list[0], edge_list[1]].reshape(-1, 1),
                                    dtype=dgl.backend.data_type_dict['float32'])
            nb_nodes = g.num_nodes()
            g.ndata['R'] = dgl.backend.tensor(data['R'][i][:nb_nodes])
            g.ndata['Z'] = dgl.backend.tensor(data['Z'][i][:nb_nodes])
            graphs.append(g)
        # ------------------------------------------

        # ---------------- NODE NUMBER FILTERING
        # ADDING CUSTOM METHODS TO FILTER AND COMPUTE ATTRIBUTES
        # Filter on number of nodes (wrapped around zip / unzip of graphs with labels)
        graphs, labels = filter_nb_nodes(graphs, self.min_node, self.max_node, list(labels))
        # ------------------------------------------

        # ---------------- COPY PASTED FROM QGNN-FINAL
        # For QM7, atomic number and coordinates are usable | Ref.: https://stackoverflow.com/a/66301026/10115198
        self.different_z = list(np.unique(data['Z']))
        for g in graphs:
            g.ndata['Z_one_hot'] = torch.stack(list(map(self.one_hot, g.ndata['Z'].numpy())))
            z = g.ndata['Z_one_hot'] if self.embed_z else g.ndata['Z'].unsqueeze(1)
            if self.use_positions:
                g.ndata['attr'] = torch.hstack((z, g.ndata['R']))
            else:
                g.ndata['attr'] = z
        # ------------------------------------------
            g.ndata['x'] = g.ndata['attr']
        targets = torch.tensor(labels).reshape((len(labels), 1))

        data_list = []
        for i, g in tqdm(enumerate(graphs[:5])):
            networkx_graph = g.to_networkx(node_attrs=['x'])
            graph = from_networkx(networkx_graph)
            graph.y = targets[i]
            data_list.append(graph)

        train_indices, val_indices, test_indices = dgl.data.utils.split_dataset(
            range(len(data_list)), frac_list=self.splits, shuffle=True, random_state=self.permutation_seed
        )
        split_dict = {
            'train': list(train_indices.indices),
            'valid': list(val_indices.indices),
            'test': list(test_indices.indices)
        }

        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]

        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]

        # torch.save(self.collate(data_list), self.processed_paths[0])
        torch.save(self.collate(data_list), self.processed_paths[0])
        torch.save(split_dict, self.processed_paths[1])

    def get_idx_split(self):
        return torch.load(self.processed_paths[1])

    # ---------------- COPY PASTED FROM QGNN-FINAL
    def one_hot(self, val):
        i = self.different_z.index(val)
        n = len(self.different_z)
        return torch.zeros(n).scatter_(0, torch.tensor([i]), 1)


# ---------------- COPY PASTED FROM QGNN-FINAL
def filter_nb_nodes(graphs, min_node, max_node, *linked_lists):
    return list(map(
        # Converting all elements from tuples to lists
        list,
        # Filtering graph & linked_data based on graph's number of nodes
        zip(*filter(
            lambda p: (min_node <= p[0].number_of_nodes()) and (p[0].number_of_nodes() <= max_node),
            zip(graphs, *linked_lists)
        ))
    ))
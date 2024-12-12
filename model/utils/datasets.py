# Copyright (c) Facebook, Inc. and its affiliates.
import pickle
import re
import numpy as np
import torch
import random

class SpatioTemporalDataset(torch.utils.data.Dataset):

    def __init__(self, train_set, test_set, train_label, test_label, train_id, test_id, train_seq, split_seq, x_dim, data, train):
        s_offset = torch.cat([torch.zeros(1, 1), torch.ones(1, x_dim)], dim=1)
        self.dataset = [torch.tensor(seq) + s_offset for seq in (train_set if train else test_set)]
        self.dataset_label = [torch.tensor(seq).float() for seq in (train_label if train else test_label)]
        self.dataset_id = [torch.tensor(int(seq)) for seq in (train_id if train else test_id)]
        self.dataset_seq = [torch.log2(torch.tensor(seq[1:])+1) for seq in (train_seq if train else split_seq)]
        self.dataset_cum_seq = [torch.log2(torch.tensor(seq).cumsum(dim=-1)[1:]) for seq in (train_seq if train else split_seq)]

    def __len__(self):
        return len(self.dataset)

    def _standardize(self, dataset):
        dataset = [torch.tensor(seq) for seq in dataset]
        full = torch.cat(dataset, dim=0)
        S = full[:, 1:]
        S_mean = S.mean(0, keepdims=True)
        S_std = S.std(0, keepdims=True)
        print(S.min(), S.max())
        return S_mean, S_std

    def unstandardize(self, x):
        return x * self.S_std + self.S_mean

    def ordered_indices(self):
        lengths = np.array([seq.shape[0] for seq in self.dataset])
        indices = np.argsort(lengths)
        return indices, lengths[indices]

    def batch_by_size(self, max_events):
        try:
            from data_utils_fast import batch_by_size_fast
        except ImportError:
            raise ImportError('Please run `python setup.py build_ext --inplace`')

        indices, num_tokens = self.ordered_indices()

        if not isinstance(indices, np.ndarray):
            indices = np.fromiter(indices, dtype=np.int64, count=-1)
        num_tokens_fn = lambda i: num_tokens[i]

        return batch_by_size_fast(
            indices, num_tokens_fn, max_tokens=max_events, max_sentences=-1, bsz_mult=1,
        )

    def __getitem__(self, index):
        return (self.dataset[index], self.dataset_label[index], self.dataset_id[index], self.dataset_seq[index], self.dataset_cum_seq[index])


class CascadeData(SpatioTemporalDataset):

    def __init__(self, args, split="train", observation_time="1800", seq_len=100, x_dim=80):
        assert split in ["train", "val", "test"]
        self.split = split
        data = args.data

        with open(f'data/{data}/train_t{observation_time}_s{seq_len}.pkl', 'rb') as ftrain:
            train_cascade, train_label, train_id = pickle.load(ftrain)
        with open(f'data/{data}/val_t{observation_time}_s{seq_len}.pkl', 'rb') as fval:
            val_cascade, val_label, val_id = pickle.load(fval)
        with open(f'data/{data}/test_t{observation_time}_s{seq_len}.pkl', 'rb') as ftest:
            test_cascade, test_label, test_id = pickle.load(ftest)

        with open(f'data/{data}/train_{observation_time}_{args.interval_num}.pkl', 'rb') as ftrain:
            train_seq = pickle.load(ftrain)
        with open(f'data/{data}/val_{observation_time}_{args.interval_num}.pkl', 'rb') as fval:
            val_seq = pickle.load(fval)
        with open(f'data/{data}/test_{observation_time}_{args.interval_num}.pkl', 'rb') as ftest:
            test_seq = pickle.load(ftest)

        dataset = {"train": train_cascade, "val": val_cascade, "test": test_cascade}
        dataset_label = {"train": train_label, "val": val_label, "test": test_label}
        dataset_id = {"train": train_id, "val": val_id, "test": test_id}
        dataset_seq = {"train": train_seq, "val": val_seq, "test": test_seq}

        train_set = train_cascade
        split_set = dataset[split]

        train_label = train_label
        split_label = dataset_label[split]

        train_id = train_id
        split_id = dataset_id[split]

        train_seq = train_seq
        split_seq = dataset_seq[split]
        super().__init__(train_set, split_set, train_label, split_label, train_id, split_id, train_seq, split_seq, x_dim, data, split == "train")

    def extra_repr(self):
        return f"Split: {self.split}"


def spatiotemporal_events_collate_fn(data):
    """Input is a list of tensors with shape (T, 1 + D)
        where T may be different for each tensor.

    Returns:
        event_times: (N, max_T)     N可以理解为batch_size
        spatial_locations: (N, max_T, D)    如果为经纬度的话，D等于2
        mask: (N, max_T)
    """
    label = [d[1] for d in data]
    label = torch.stack(label, dim=0)

    id = [d[2] for d in data]
    id = torch.stack(id, dim=0)

    seq = [d[3] for d in data]
    seq = torch.stack(seq, dim=0)

    cum_seq = [d[3] for d in data]
    cum_seq = torch.stack(cum_seq, dim=0)

    data = [d[0] for d in data]
    if len(data) == 0:
        # Dummy batch, sometimes this occurs when using multi-GPU.
        return torch.zeros(1, 1), torch.zeros(1, 1, 2), torch.zeros(1, 1)
    dim = data[0].shape[1]
    lengths = [seq.shape[0] for seq in data]
    max_len = max(lengths)
    padded_seqs = [torch.cat([s, torch.zeros(max_len - s.shape[0], dim).to(s)], 0) if s.shape[0] != max_len else s for s in data]
    data = torch.stack(padded_seqs, dim=0)
    event_times = data[:, :, 0]
    spatial_locations = data[:, :, 1:]
    mask = torch.stack([torch.cat([torch.ones(seq_len), torch.zeros(max_len - seq_len)], dim=0) for seq_len in lengths])

    return event_times, spatial_locations, mask, label, seq, cum_seq, id

# Copyright (c) Facebook, Inc. and its affiliates.

import abc
import torch
import torch.nn as nn
import torch.nn.functional as F


class TemporalPointProcess(nn.Module):
    __metaclass__ = abc.ABCMeta

    @abc.abstractmethod
    def logprob(self, event_times, input_mask, t0, t1):
        """
        Args:
            event_times: (N, T)
            input_mask: (N, T)
            t0: (N,) or (1,)
            t1: (N,) or (1,)
        """
        raise NotImplementedError



def lowtri(A):
    return torch.tril(A, diagonal=-1)


def fill_triu(A, value):
    A = lowtri(A)
    A = A + torch.triu(torch.ones_like(A)) * value
    return A

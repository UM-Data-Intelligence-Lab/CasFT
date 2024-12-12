# Copyright (c) Facebook, Inc. and its affiliates.

from abc import ABCMeta, abstractmethod
import torch
import torch.nn as nn
import torch.nn.functional as F
from model.diffusion import GaussianDiffusion_ST, Model_all, ST_Diffusion

class Predictor(nn.Module):
    """ Prediction of next event type. """

    def __init__(self, dim, dim_out=1):
        super().__init__()

        self.linear1 = nn.Linear(dim, 2*dim)
        self.linear2 = nn.Linear(dim*2, dim)
        self.linear3 = nn.Linear(dim, dim_out)

    def forward(self, data):
        out1 = F.relu(self.linear1(data))
        out2 = self.linear2(out1)
        out3 = self.linear3(out2)
        out = F.softplus(out3)
        return out


class Classifier(nn.Module):
    """ Prediction of next event type. """

    def __init__(self, dim, num_types):
        super().__init__()

        self.linear = nn.Linear(dim, num_types)
        nn.init.xavier_normal_(self.linear.weight)

    def forward(self, data):
        out = self.linear(data)
        return out


class SpatiotemporalModel(nn.Module, metaclass=ABCMeta):

    @abstractmethod
    def forward(self, event_times, spatial_events, input_mask, t0, t1, label, seq, cum_seq):
        """
        Args:
            event_times: (N, T)
            spatial_events: (N, T, D)
            input_mask: (N, T)
            t0: () or (N,)
            t1: () or (N,)
        """
        pass


class CombinedSpatiotemporalModel5(SpatiotemporalModel):
#   ode_trans_mlp
    def __init__(self, temporal_model, encoder_model, args):
        super().__init__()
        self.encoder = encoder_model
        self.temporal_model = temporal_model
        self.args = args

        model = ST_Diffusion(
            cond_dim=int(self.args.tpp_hdims) + self.args.interval_num,
            output_dim=self.args.interval_num
        )

        # diffusion模型，包括了ST_diffusion
        diffusion = GaussianDiffusion_ST(
            model,
            loss_type=self.args.loss_type,
            seq_length=self.args.interval_num,
            timesteps=self.args.timesteps,
            sampling_timesteps=self.args.sampling_timesteps,
            objective=self.args.objective,
            beta_schedule='cosine'
        )

        self.Model = Model_all(None, diffusion)

        self.pred = Predictor(dim=self.temporal_model.hdim + self.args.interval_num)

        num_units = self.args.interval_num
        self.seq_feature = nn.Sequential(
                nn.Linear(num_units, num_units*2),
                nn.ReLU(),
                nn.Linear(num_units*2, num_units*2),
                nn.ReLU(),
                nn.Linear(num_units*2, num_units)
        )


    def forward(self, event_times, spatial_events, input_mask, t0, t1, label, seq, cum_seq):
        event_times = event_times * float(self.args.observation_time) / float(self.args.prediction_time)
        if self.encoder:
            spatial_events = self.encoder(spatial_events, event_times, input_mask)
        intensities, Lamdas, prejump_hidden_states = self._temporal_logprob(event_times, spatial_events, input_mask, t0,
                                                                            t1)
        # (b, seq_len), list: 每一个(b,); (b, seq_len, dim)
        all_Lamdas = torch.stack(Lamdas, dim=1)  # (N, T)

        time_emb = prejump_hidden_states[:, 0, :]
        _, loss = self.Model.diffusion(seq.unsqueeze(1), torch.cat([time_emb.unsqueeze(1), all_Lamdas[:, 1:].unsqueeze(1)], dim=-1))
        sampled_seq = self.Model.diffusion.sample(batch_size=event_times.shape[0], cond=torch.cat([time_emb.unsqueeze(1), all_Lamdas[:, 1:].unsqueeze(1)], dim=-1))
        if self.args.FC:
            sampled_seq = self.seq_feature(sampled_seq)

        pre_label = self.pred(torch.cat([time_emb, sampled_seq.squeeze(1)], dim=1))
        return pre_label, loss * self.args.loss_scale

    def _temporal_logprob(self, event_times, spatial_events, input_mask, t0, t1):
        return self.temporal_model.logprob(event_times, spatial_events, input_mask, t0, t1)


def zero_diffeq(t, h):
    return torch.zeros_like(h)


def get_non_pad_mask(seq):
    """ Get the non-padding positions. """
    assert seq.dim() == 2
    return seq.ne(0).type(torch.float).unsqueeze(-1)


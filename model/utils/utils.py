# Copyright (c) Facebook, Inc. and its affiliates.
import math
import datetime
import torch.distributed as dist
import logging
import os
import torch
import psutil
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error

class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class RunningAverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self, momentum=0.99):
        self.momentum = momentum
        self.reset()

    def reset(self):
        self.val = None
        self.avg = 0

    def update(self, val):
        if self.val is None:
            self.avg = val
        else:
            self.avg = self.avg * self.momentum + val * (1 - self.momentum)
        self.val = val


def makedirs(dirname):
    os.makedirs(dirname, exist_ok=True)


def get_logger(logpath, package_files=[], displaying=True, saving=True, debug=False):
    logger = logging.getLogger()
    if debug:
        level = logging.DEBUG
    else:
        level = logging.INFO

    if (logger.hasHandlers()):
        logger.handlers.clear()

    logger.setLevel(level)
    formatter = logging.Formatter('%(asctime)s - %(message)s')
    if saving:
        info_file_handler = logging.FileHandler(logpath, mode="a")
        info_file_handler.setLevel(level)
        info_file_handler.setFormatter(formatter)
        logger.addHandler(info_file_handler)
    if displaying:
        console_handler = logging.StreamHandler()
        console_handler.setLevel(level)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

    for f in package_files:
        logger.info(f)
        with open(f, "r") as package_f:
            logger.info(package_f.read())

    return logger


def save_checkpoint(state, savedir, itr, last_checkpoints=None, num_checkpoints=None):
    if not os.path.exists(savedir):
        os.makedirs(savedir)
    filename = os.path.join(savedir, 'checkpt-%08d.pth' % itr)
    torch.save(state, filename)

    if last_checkpoints is not None and num_checkpoints is not None:
        last_checkpoints.append(itr)
        if len(last_checkpoints) > num_checkpoints:
            rm_itr = last_checkpoints.pop(0)
            old_checkpt = os.path.join(savedir, 'checkpt-%08d.pth' % rm_itr)
            if os.path.exists(old_checkpt):
                os.remove(old_checkpt)


def find_latest_checkpoint(savedir):
    import glob
    import re

    checkpt_files = glob.glob(os.path.join(savedir, 'checkpt-[0-9]*.pth'))

    if not checkpt_files:
        return None

    def extract_itr(f):
        s = re.findall('(\d+).pth$', f)
        return int(s[0]) if s else -1

    latest_itr = max(checkpt_files, key=extract_itr)
    return latest_itr


def inf_generator(iterable):
    """Allows training with DataLoaders in a single infinite loop:
        for i, (x, y) in enumerate(inf_generator(train_loader)):
    """
    iterator = iterable.__iter__()
    while True:
        try:
            yield iterator.__next__()
        except StopIteration:
            iterator = iterable.__iter__()


class ExponentialMovingAverage(object):

    def __init__(self, module, decay=0.999):
        """Initializes the model when .apply() is called the first time.
        This is to take into account data-dependent initialization that occurs in the first iteration."""
        self.decay = decay
        self.module_params = {n: p for (n, p) in module.named_parameters()}
        self.ema_params = {n: p.data.clone() for (n, p) in module.named_parameters()}
        self.nparams = sum(p.numel() for (_, p) in self.ema_params.items())

    def apply(self, decay=None):
        decay = decay or self.decay
        with torch.no_grad():
            for name, param in self.module_params.items():
                self.ema_params[name] -= (1 - decay) * (self.ema_params[name] - param.data)

    def set(self, named_params):
        with torch.no_grad():
            for name, param in named_params.items():
                self.ema_params[name].copy_(param)

    def replace_with_ema(self):
        for name, param in self.module_params.items():
            param.data.copy_(self.ema_params[name])

    def swap(self):
        for name, param in self.module_params.items():
            tmp = self.ema_params[name].clone()
            self.ema_params[name].copy_(param.data)
            param.data.copy_(tmp)

    def __repr__(self):
        return (
            '{}(decay={}, module={}, nparams={})'.format(
                self.__class__.__name__, self.decay, self.module.__class__.__name__, self.nparams
            )
        )


def get_msle(prediction, event_popularity):
    """ popularity prediction loss. """
    true = torch.unsqueeze(event_popularity, 1)
    true = true + 1
    prediction = prediction + 1

    true = torch.log2(true)
    prediction = torch.log2(prediction)

    diff = prediction - true
    msle = diff * diff

    return msle


def get_smape(prediction, event_popularity):
    true = torch.unsqueeze(event_popularity, 1)
    true = true + 2
    prediction = prediction + 2

    true = torch.log2(true)
    prediction = torch.log2(prediction)

    diff = prediction - true
    smape = torch.abs(diff) / true

    return smape

def msle(pred, label):
    pred = np.log2(pred + 1)
    label = np.log2(label + 1)
    return np.around(mean_squared_error(label, pred, multioutput='raw_values'), 4)[0]

def pcc(pred, label):
    pred = np.log2( pred + 1)
    label = np.log2(label + 1)
    pred_mean, label_mean = np.mean(pred, axis=0), np.mean(label, axis=0)
    pre_std, label_std = np.std(pred, axis=0), np.std(label, axis=0)
    return np.around(np.mean((pred - pred_mean) * (label - label_mean) / (pre_std * label_std), axis=0), 4)

def male(pred, label):
    pred = np.log2( pred + 1)
    label = np.log2(label + 1)
    return np.around(mean_absolute_error(label, pred, multioutput='raw_values'), 4)[0]

def mape(pred, label):
    result = np.mean(np.abs(np.log2(pred + 2) - np.log2(label + 2)) / np.log2(label + 2))
    return np.around(result, 4)

def smape(pred, label):
    result = 2 * np.mean(np.abs(np.log2(pred + 2) - np.log2(label + 2)) / (np.log2(label + 2) + np.log2(pred + 2)))
    return np.around(result, 4)

def r2(pred, label):
    pred = np.log2( pred + 1)
    label = np.log2(label + 1)
    label_mean = np.mean(label, axis=0)
    r2 = 1 - np.mean(np.square(pred - label), axis=0) / np.mean(np.square(label_mean - label), axis=0)
    return np.around(r2, 4)

class Metric:
    def __init__(self):
        self.cascade_popularity = {'pred': [], 'label': [], 'id': [], 'observed': []}
        self.all_metric = {'msle': 0, 'male': 0, 'pcc': 0, 'mape': 0, 'smape': 0, 'r2': 0}

    def update(self, pred, label, id, observed):
        self.cascade_popularity['pred'].extend(pred)
        self.cascade_popularity['label'].extend(label)
        self.cascade_popularity['id'].extend(id)
        self.cascade_popularity['observed'].extend(observed)

    def calculate_metric(self):
        pred_csv = {'id': self.cascade_popularity['id'], 'observed': self.cascade_popularity['observed'],
                    'label': self.cascade_popularity['label'],
                    'pred': self.cascade_popularity['pred']}
        self.cascade_popularity['pred'] = np.array(self.cascade_popularity['pred'])
        self.cascade_popularity['label'] = np.array(self.cascade_popularity['label'])

        self.all_metric['msle'] = msle(self.cascade_popularity['pred'], self.cascade_popularity['label'])
        self.all_metric['male'] = male(self.cascade_popularity['pred'], self.cascade_popularity['label'])
        self.all_metric['pcc'] = pcc(self.cascade_popularity['pred'], self.cascade_popularity['label'])
        self.all_metric['mape'] = mape(self.cascade_popularity['pred'], self.cascade_popularity['label'])
        self.all_metric['smape'] = smape(self.cascade_popularity['pred'], self.cascade_popularity['label'])
        self.all_metric['r2'] = r2(self.cascade_popularity['pred'], self.cascade_popularity['label'])

        return self.all_metric, pred_csv


def setup(rank, world_size, port):
    #ncll的ip和port
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = str(port)

    # initialize the process group
    dist.init_process_group("nccl", rank=rank, world_size=world_size, timeout=datetime.timedelta(minutes=30))


def cleanup():
    dist.destroy_process_group()


def memory_usage_psutil():
    # return the memory usage in MB
    process = psutil.Process(os.getpid())
    mem = process.memory_info()[0] / float(2 ** 20)
    return mem


def cosine_decay(learning_rate, global_step, decay_steps, alpha=0.0):
    global_step = min(global_step, decay_steps)
    cosine_decay = 0.5 * (1 + math.cos(math.pi * global_step / decay_steps))
    decayed = (1 - alpha) * cosine_decay + alpha
    return learning_rate * decayed


def learning_rate_schedule(global_step, warmup_steps, base_learning_rate, train_steps):
    warmup_steps = int(round(warmup_steps))
    scaled_lr = base_learning_rate
    if warmup_steps:
        learning_rate = global_step / warmup_steps * scaled_lr
    else:
        learning_rate = scaled_lr

    if global_step < warmup_steps:
        learning_rate = learning_rate
    else:
        learning_rate = cosine_decay(scaled_lr, global_step - warmup_steps, train_steps - warmup_steps)
    return learning_rate


def set_learning_rate(optimizer, lr):
    for i, group in enumerate(optimizer.param_groups):
        group['lr'] = lr


def cast(tensor, device):
    return tensor.float().to(device) if torch.is_tensor(tensor) else None


def to_numpy(x):
    if torch.is_tensor(x):
        return x.cpu().detach().numpy()
    return [to_numpy(x_i) for x_i in x]


def get_t0_t1(t1, args):
    data_split = []
    t_start_o = float(args.observation_time) / float(args.prediction_time)
    t_end_p = 1
    data_split.append(t_start_o)
    interval = (t_end_p - t_start_o) / float(args.interval_num)
    for _ in range(args.interval_num - 1):
        data_split.append(data_split[-1] + interval)
    data_split.append(t_end_p)

    if t1:
        return torch.tensor([0.0]), torch.tensor(data_split)
    else:
        return torch.tensor([0.0]), None

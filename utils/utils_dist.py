# Copyright (c) 2015-present, Facebook, Inc.
# All rights reserved.
"""
Misc functions, including distributed helpers.
Mostly copy-paste from torchvision references.
"""
import io
import os
import time
from collections import defaultdict, deque
import datetime

import math
import torch
import torch.distributed as dist


class ListerLoss():
    '''loss calculator for LISTER NeighborDecoder Module'''

    def __init__(self):
        self.celoss_fn = torch.nn.CrossEntropyLoss(reduction='none')
        self.coef = (1.0, 0.01, 0.001)  # for loss of rec, eos and ent respectively

    def calc_rec_loss(self, logits, targets, target_lens, mask):
        """
        Args:
            logits: [minibatch, C, T], not passed to the softmax func.
            targets, torch.cuda.LongTensor [minibatch, T]
            target_lens: [minibatch]
            mask: [minibatch, T]
        """
        losses = self.celoss_fn(logits, targets)
        losses = losses * mask
        loss = losses.sum(-1) / (target_lens + 1e-10)
        loss = loss.mean()
        return loss

    def calc_eos_loc_loss(self, char_maps, target_lens, eps=1e-10):
        max_tok = char_maps.shape[2]
        eos_idx = (target_lens - 1).contiguous().view(-1, 1, 1).expand(-1, 1, max_tok).to(torch.int64)
        # import IPython;IPython.embed()
        eos_maps = torch.gather(char_maps, dim=1, index=eos_idx).squeeze(1)  # (b, max_tok)
        loss = (eos_maps[:, -1] + eps).log().neg()
        return loss.mean()

    def calc_entropy(self, p: torch.Tensor, mask: torch.Tensor, eps=1e-10):
        """
        Args:
            p: probability distribution over the last dimension, of size (..., L, C)
            mask: (..., L)
        """
        p_nlog = (p + eps).log().neg()
        ent = p * p_nlog
        ent = ent.sum(-1) / math.log(p.size(-1) + 1)
        ent = (ent * mask).sum(-1) / (mask.sum(-1) + eps)  # (...)
        ent = ent.mean()
        # ent.fill_(0)
        return ent

    def get_loss(self, model_output, labels, label_lens):
        batch_size, max_len = labels.size()
        seq_range = torch.arange(0, max_len, device=labels.device).long().unsqueeze(0).expand(
            batch_size, max_len)
        seq_len = label_lens.unsqueeze(1).expand_as(seq_range)
        mask = (seq_range < seq_len).float()  # [batch_size, max_len]

        l_rec, l_eos, l_ent = [], [], []
        iters = len(model_output['logits'])
        for i in range(iters):
            l_rec.append(self.calc_rec_loss(model_output['logits'][i].transpose(1, 2), labels, label_lens, mask))
            l_eos.append(self.calc_eos_loc_loss(model_output['char_maps'][i], label_lens))
            l_ent.append(self.calc_entropy(model_output['char_maps'][i], mask))
        if all([li.item() > 2.1 for li in l_rec]):  # avoid hard training in the start
            l_rec = l_rec[0] + sum(l_rec[1:]) / (iters - 1 + 1e-8) * 0.0
            l_eos = l_eos[0] + sum(l_eos[1:]) / (iters - 1 + 1e-8) * 0.0
            l_ent = l_ent[0] + sum(l_ent[1:]) / (iters - 1 + 1e-8) * 0.0
        else:
            l_rec = sum(l_rec) / iters
            l_eos = sum(l_eos) / iters
            l_ent = sum(l_ent) / iters

        loss = l_rec * self.coef[0] + l_eos * self.coef[1] + l_ent * self.coef[2]
        loss_dict = dict(
            loss=loss,
            l_rec=l_rec,
            l_eos=l_eos,
            l_ent=l_ent,
        )
        return loss_dict


class SmoothedValue(object):
    """Track a series of values and provide access to smoothed values over a
    window or the global series average.
    """

    def __init__(self, window_size=20, fmt=None):
        if fmt is None:
            fmt = "{median:.4f} ({global_avg:.4f})"
        self.deque = deque(maxlen=window_size)
        self.total = 0.0
        self.count = 0
        self.fmt = fmt

    def update(self, value, n=1):
        self.deque.append(value)
        self.count += n
        self.total += value * n

    def synchronize_between_processes(self):
        """
        Warning: does not synchronize the deque!
        """
        if not is_dist_avail_and_initialized():
            return
        t = torch.tensor([self.count, self.total], dtype=torch.float64, device='cuda')
        dist.barrier()
        dist.all_reduce(t)
        t = t.tolist()
        self.count = int(t[0])
        self.total = t[1]

    @property
    def median(self):
        d = torch.tensor(list(self.deque))
        return d.median().item()

    @property
    def avg(self):
        d = torch.tensor(list(self.deque), dtype=torch.float32)
        return d.mean().item()

    @property
    def global_avg(self):
        return self.total / self.count

    @property
    def max(self):
        return max(self.deque)

    @property
    def value(self):
        return self.deque[-1]

    def __str__(self):
        return self.fmt.format(
            median=self.median,
            avg=self.avg,
            global_avg=self.global_avg,
            max=self.max,
            value=self.value)


class MetricLogger(object):
    def __init__(self, delimiter="\t"):
        self.meters = defaultdict(SmoothedValue)
        self.delimiter = delimiter

    def update(self, **kwargs):
        for k, v in kwargs.items():
            if isinstance(v, torch.Tensor):
                v = v.item()
            assert isinstance(v, (float, int))
            self.meters[k].update(v)

    def __getattr__(self, attr):
        if attr in self.meters:
            return self.meters[attr]
        if attr in self.__dict__:
            return self.__dict__[attr]
        raise AttributeError("'{}' object has no attribute '{}'".format(
            type(self).__name__, attr))

    def __str__(self):
        loss_str = []
        for name, meter in self.meters.items():
            loss_str.append(
                "{}: {}".format(name, str(meter))
            )
        return self.delimiter.join(loss_str)

    def synchronize_between_processes(self):
        for meter in self.meters.values():
            meter.synchronize_between_processes()

    def add_meter(self, name, meter):
        self.meters[name] = meter

    def log_every(self, iterable, print_freq, header=None):
        i = 0
        if not header:
            header = ''
        start_time = time.time()
        end = time.time()
        iter_time = SmoothedValue(fmt='{avg:.4f}')
        data_time = SmoothedValue(fmt='{avg:.4f}')
        space_fmt = ':' + str(len(str(len(iterable)))) + 'd'
        log_msg = [
            header,
            '[{0' + space_fmt + '}/{1}]',
            'eta: {eta}',
            '{meters}',
            'time: {time}',
            'data: {data}'
        ]
        if torch.cuda.is_available():
            log_msg.append('max mem: {memory:.0f}')
        log_msg = self.delimiter.join(log_msg)
        MB = 1024.0 * 1024.0
        for obj in iterable:
            data_time.update(time.time() - end)
            yield obj
            iter_time.update(time.time() - end)
            if i % print_freq == 0 or i == len(iterable) - 1:
                eta_seconds = iter_time.global_avg * (len(iterable) - i)
                eta_string = str(datetime.timedelta(seconds=int(eta_seconds)))
                if torch.cuda.is_available():
                    print(log_msg.format(
                        i, len(iterable), eta=eta_string,
                        meters=str(self),
                        time=str(iter_time), data=str(data_time),
                        memory=torch.cuda.max_memory_allocated() / MB))
                else:
                    print(log_msg.format(
                        i, len(iterable), eta=eta_string,
                        meters=str(self),
                        time=str(iter_time), data=str(data_time)))
            i += 1
            end = time.time()
        total_time = time.time() - start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        print('{} Total time: {} ({:.4f} s / it)'.format(
            header, total_time_str, total_time / len(iterable)))


def _load_checkpoint_for_ema(model_ema, checkpoint):
    """
    Workaround for ModelEma._load_checkpoint to accept an already-loaded object
    """
    mem_file = io.BytesIO()
    torch.save(checkpoint, mem_file)
    mem_file.seek(0)
    model_ema._load_checkpoint(mem_file)


def setup_for_distributed(is_master):
    """
    This function disables printing when not in master process
    """
    import builtins as __builtin__
    builtin_print = __builtin__.print

    def print(*args, **kwargs):
        force = kwargs.pop('force', False)
        if is_master or force:
            builtin_print(*args, **kwargs)

    __builtin__.print = print


def is_dist_avail_and_initialized():
    if not dist.is_available():
        return False
    if not dist.is_initialized():
        return False
    return True


def get_world_size():
    if not is_dist_avail_and_initialized():
        return 1
    return dist.get_world_size()


def get_rank():
    if not is_dist_avail_and_initialized():
        return 0
    return dist.get_rank()


def is_main_process():
    return get_rank() == 0


def save_on_master(*args, **kwargs):
    if is_main_process():
        torch.save(*args, **kwargs)

def init_distributed_mode(args):
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        args.rank = int(os.environ["RANK"])
        args.world_size = int(os.environ['WORLD_SIZE'])
        args.gpu = int(os.environ['LOCAL_RANK'])
    elif 'SLURM_PROCID' in os.environ:
        args.rank = int(os.environ['SLURM_PROCID'])
        args.gpu = args.rank % torch.cuda.device_count()
    else:
        print('Not using distributed mode')
        args.distributed = False
        return

    args.distributed = True

    torch.cuda.set_device(args.gpu)
    args.dist_backend = 'nccl'
    print('| distributed init (rank {}): {}'.format(
        args.rank, args.dist_url), flush=True)
    dist.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                            world_size=args.world_size, rank=args.rank)
    dist.barrier()
    setup_for_distributed(args.rank == 0)

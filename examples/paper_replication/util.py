from collections import namedtuple

import torch


class AverageMeter(object):
    """
    Computes and stores the average and current value
    Copied from: https://github.com/pytorch/examples/blob/master/imagenet/main.py
    """
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


class Welford(object):
    """
    Computes and stores a running average and variance
    """
    def __init__(self):
        self.reset()

    def reset(self):
        self._count = 0
        self._mean = None
        self._sum_sq = None

    # for a new value newValue, compute the new count, new mean, the new M2.
    # mean accumulates the mean of the entire dataset
    # M2 aggregates the squared distance from the mean
    # count aggregates the number of samples seen so far
    def update(self, new_value, batch=True):
        if isinstance(new_value, torch.autograd.Variable):
            new_value = new_value.data
        if not batch:
            new_value = new_value.unsqueeze(0)

        self._mean = new_value.new(
            *list(new_value.size())[1:]).zero_() if self._mean is None else self._mean
        self._sum_sq = new_value.new(
            *list(new_value.size())[1:]).zero_() if self._sum_sq is None else self._sum_sq

        for item in new_value:
            self._count += 1
            delta = item - self._mean
            self._mean += (item - self._mean) / float(self._count)
            self._sum_sq += delta * (item - self._mean)

    @property
    def mean(self):
        return self._mean

    @property
    def var(self):
        return self._sum_sq / (self._count - 1)

    @property
    def std(self):
        return self.var.sqrt()


def result_class(fields):
    class Result(namedtuple('Result', fields)):
        def items(self):
            for field in self._fields:
                yield (field, getattr(self, field))

        def to_str(self):
            return ",".join(str(item) for item in self)

        def __repr__(self):
            res = 'Results:\n'
            fieldstrs = []
            for key in self._fields:
                fieldstrs.append('  - %s: %s' % (key, repr(getattr(self, key))))
            res = res + '\n'.join(fieldstrs)
            return res

    return Result


def output_class(fields):
    class Output(namedtuple('Output', fields)):
        def __repr__(self):
            res = 'Outputs:\n'
            fieldstrs = []
            for key in self._fields:
                fieldstrs.append('  - %s: %s' % (key, repr(getattr(self, key).size())))
            res = res + '\n'.join(fieldstrs)
            return res

    return Output

"""Decide, without trying both, whether folding a dispatch site pays.

Two numbers drive the decision:

  miss rate     how often an indirect branch predictor would mispredict the
                next handler for this key stream. Estimated by simulating a
                TAGE style predictor over the stream, because what matters is
                not how skewed the key histogram is but how predictable the
                sequence is. A stream of long runs is trivially predictable
                even with hundreds of equally frequent cases.

  repeat rate   how often consecutive keys land in the same bucket, which is
                what makes the folded kernel's read modify write histogram
                stall on store forwarding.

Both are combined with costs calibrated on the host CPU, so the crossover is
a measured property of the machine rather than a constant baked into a table.
"""
import random

import numpy as np

from . import lowering as L
from .jit import Buffer, Kernel, time_kernel

SAMPLE = 1 << 16


class TagePredictor(object):
    """A small TAGE style indirect target predictor.

    One table per history length, longest matching history wins, mispredicts
    allocate into a longer table. This is the structure every modern x86 core
    uses for indirect branches, and it is why a correlated key stream can be
    predicted almost perfectly even when the key histogram looks flat.
    """

    def __init__(self, history_lengths=(0, 2, 4, 8, 16), capacity=1 << 12):
        self.lengths = history_lengths
        self.capacity = capacity
        self.tables = [{} for _ in history_lengths]

    def _keys(self, history):
        out = []
        for i, ln in enumerate(self.lengths):
            if ln == 0:
                out.append(0)
            elif len(history) < ln:
                out.append(None)
            else:
                out.append(hash(tuple(history[-ln:])) & 0xFFFFFFFF)
        return out

    def run(self, stream):
        history = []
        misses = 0
        total = 0
        for key in stream:
            idxs = self._keys(history)
            provider = None
            predicted = None
            for i in range(len(self.lengths) - 1, -1, -1):
                if idxs[i] is None:
                    continue
                hit = self.tables[i].get(idxs[i])
                if hit is not None:
                    provider, predicted = i, hit[0]
                    break
            total += 1
            if predicted != key:
                misses += 1
                # correct the provider, then allocate one table longer
                if provider is not None:
                    tgt, ctr = self.tables[provider][idxs[provider]]
                    if ctr <= 0:
                        self.tables[provider][idxs[provider]] = (key, 1)
                    else:
                        self.tables[provider][idxs[provider]] = (tgt, ctr - 1)
                start = 0 if provider is None else provider + 1
                for i in range(start, len(self.lengths)):
                    if idxs[i] is None:
                        continue
                    tbl = self.tables[i]
                    if len(tbl) >= self.capacity:
                        tbl.pop(next(iter(tbl)))
                    tbl[idxs[i]] = (key, 1)
                    break
            elif provider is not None:
                tgt, ctr = self.tables[provider][idxs[provider]]
                if ctr < 3:
                    self.tables[provider][idxs[provider]] = (tgt, ctr + 1)
            history.append(key)
            if len(history) > 16:
                del history[0]
        return misses / float(max(1, total))


def miss_rate(stream, sample=SAMPLE):
    return TagePredictor().run(stream[:sample])


def repeat_rate(stream, sample=SAMPLE):
    s = stream[:sample]
    if len(s) < 2:
        return 0.0
    same = sum(1 for i in range(1, len(s)) if s[i] == s[i - 1])
    return same / float(len(s) - 1)


class CostModel(object):
    """Linear cost models for the direct and the folded kernel, in ns per key."""

    def __init__(self, direct, folded, residual):
        self.c_hit, self.penalty = direct
        self.f_item, self.f_repeat, self.f_bucket = folded
        self.residual = residual

    def direct_ns(self, miss):
        return self.c_hit + self.penalty * miss

    def fold_ns(self, repeat, n_slots, block=L.DEFAULT_BLOCK):
        return (self.f_item + self.f_repeat * repeat
                + self.f_bucket * n_slots / float(block))

    def decide(self, stream, n_slots, block=L.DEFAULT_BLOCK):
        miss = miss_rate(stream)
        rep = repeat_rate(stream)
        d = self.direct_ns(miss)
        f = self.fold_ns(rep, n_slots, block)
        return {
            'miss_rate': miss,
            'repeat_rate': rep,
            'predicted_direct_ns': d,
            'predicted_fold_ns': f,
            'choice': 'fold' if f < d else 'table',
            'predicted_speedup': d / f,
        }

    def describe(self):
        return ('direct: %.2f ns + %.2f ns per mispredict\n'
                'folded: %.2f ns + %.2f ns per repeat + %.2f ns per bucket '
                'per block\nfit residual: %.3f ns' %
                (self.c_hit, self.penalty, self.f_item, self.f_repeat,
                 self.f_bucket, self.residual))

    def as_dict(self):
        return {'c_hit_ns': self.c_hit, 'mispredict_ns': self.penalty,
                'fold_item_ns': self.f_item, 'fold_repeat_ns': self.f_repeat,
                'fold_bucket_ns': self.f_bucket, 'fit_residual_ns': self.residual}


def _calibration_streams(n, rng):
    """Synthetic streams used only to measure this machine's constants.

    They are chosen to spread out over miss rate and repeat rate: uniform
    streams mispredict on nearly every key, cyclic and run structured streams
    are almost free, and the skewed ones land in between.
    """
    out = []

    def uniform(k):
        return [rng.randrange(k) for _ in range(n)]

    out.append(('uniform256', uniform(256), 256))
    out.append(('uniform64', uniform(64), 64))
    out.append(('uniform16', uniform(16), 16))
    out.append(('uniform4', uniform(4), 4))

    w = [1.0 / (i + 1) ** 1.2 for i in range(64)]
    out.append(('zipf64', rng.choices(range(64), weights=w, k=n), 64))
    for cases, p in ((32, 0.90), (8, 0.99)):
        wt = [p] + [(1.0 - p) / (cases - 1)] * (cases - 1)
        out.append(('hot%d' % int(p * 100), rng.choices(range(cases),
                                                        weights=wt, k=n), cases))
    out.append(('cycle7', [i % 7 for i in range(n)], 7))

    runs = []
    while len(runs) < n:
        v = rng.randrange(64)
        runs.extend([v] * rng.randrange(4, 13))
    out.append(('runs64', runs[:n], 64))
    return out


def calibrate(n=1 << 17, block=L.DEFAULT_BLOCK, seed=20251015, verbose=True):
    """Measure this CPU's dispatch constants and fit both cost models."""
    rng = random.Random(seed)
    rows = []
    if verbose:
        print('calibrating on this CPU (%d keys per point)' % n)
        print('  %-11s %6s %6s  %8s %8s' %
              ('stream', 'miss', 'repeat', 'table ns', 'fold ns'))
    for name, stream, cases in _calibration_streams(n, rng):
        site = L.Site(range(cases), 0, cases - 1)
        ref = site.reference(stream)
        buf = Buffer(stream, L.scratch_words(site, block))
        measured = {}
        for kind, builder in (('table', L.jump_table(site)),
                              ('fold', L.fold(site, block))):
            kern = Kernel(kind, builder)
            ns, acc = time_kernel(kern, buf, reps=9)
            if acc != ref:
                raise AssertionError('%s kernel disagreed on %s' % (kind, name))
            measured[kind] = ns
            kern.free()
        miss = miss_rate(stream)
        rep = repeat_rate(stream)
        rows.append((name, miss, rep, site.n_slots,
                     measured['table'], measured['fold']))
        if verbose:
            print('  %-11s %6.3f %6.3f  %8.2f %8.2f'
                  % (name, miss, rep, measured['table'], measured['fold']))

    a_direct = np.array([[1.0, r[1]] for r in rows])
    b_direct = np.array([r[4] for r in rows])
    direct, res_d, _, _ = np.linalg.lstsq(a_direct, b_direct, rcond=None)

    a_fold = np.array([[1.0, r[2], r[3] / float(block)] for r in rows])
    b_fold = np.array([r[5] for r in rows])
    folded, res_f, _, _ = np.linalg.lstsq(a_fold, b_fold, rcond=None)

    err = np.concatenate([a_direct.dot(direct) - b_direct,
                          a_fold.dot(folded) - b_fold])
    residual = float(np.sqrt((err ** 2).mean()))
    model = CostModel(tuple(direct), tuple(folded), residual)
    if verbose:
        print(model.describe())
    return model, rows

"""Code generators for the dispatch lowerings being compared.

A dispatch site is described by a Site: the set of key values that have their
own handler, the key domain, and how much work each handler does. Each
lowering turns a Site into a complete benchmark loop.

The direct lowerings (chain, binary search, jump table, peeled table) are the
shapes a production compiler emits for a switch. The folded lowering is the
one this project is about: it sorts a block of items by key with two
branchless counting-sort passes and then runs each handler over its own
contiguous run of items, so the only branch left is a loop-back that the
predictor always gets right.
"""
from .x64 import Asm, mem

HANDLER_STEP = 0x9E3779B9     # arbitrary odd constant for filler work


class Site(object):
    def __init__(self, keys, domain_lo, domain_hi, body_ops=0, seed=0x2545F491):
        self.keys = sorted(set(keys))
        self.lo = domain_lo
        self.hi = domain_hi
        self.body_ops = body_ops
        # handler constants stay below 2**31 so that the 32-bit adds used by
        # the handlers and the sign-extending 64-bit add used by the folded
        # kernel agree bit for bit
        self.const = {k: (((k + 1) * seed) & 0x7FFFFFFF) for k in self.keys}
        self.default_const = 0x0D0D0D0D

    @property
    def n_slots(self):
        """Buckets for the folded kernel: one per domain value plus default."""
        return (self.hi - self.lo + 1) + 1

    def handler_value(self, key):
        c = self.const.get(key, self.default_const)
        return (c + self.body_ops * HANDLER_STEP) & 0xFFFFFFFF

    def reference(self, stream):
        total = 0
        for i, k in enumerate(stream):
            total += self.handler_value(k) + i
        return total & 0xFFFFFFFFFFFFFFFF


# --------------------------------------------------------------------- direct

def _handler_body(a, site, key, result='r9'):
    a.mov_rm_imm32(result, site.const.get(key, site.default_const), w=False)
    for _ in range(site.body_ops):
        a.add_rm_imm32(result, HANDLER_STEP, w=False)


def _direct_loop(site, emit_dispatch, order=None):
    """keys in rcx, n in rdx. rbx is the item index, r8 the key, r9 the result."""
    a = Asm()
    a.push('rbx')
    a.zero('rax')
    a.zero('rbx')
    a.test_r_rm('rdx', 'rdx', w=True)
    a.jcc('e', 'done')
    a.label('loop')
    a.mov_r_rm('r8', mem('rcx', 'rbx', 4, 0), w=False)
    emit_dispatch(a)
    for key in (order if order is not None else site.keys):
        a.label('h%d' % key)
        _handler_body(a, site, key)
        a.jmp('join')
    a.label('hdefault')
    _handler_body(a, site, None)
    a.label('join')
    a.add_r_rm('rax', 'r9', w=True)
    a.add_r_rm('rax', 'rbx', w=True)
    a.inc_rm('rbx', w=True)
    a.cmp_rm_r('rbx', 'rdx', w=True)
    a.jcc('b', 'loop')
    a.label('done')
    a.pop('rbx')
    a.ret()
    return a


def chain(site, order=None):
    """if (k==a) ... else if (k==b) ... : one compare and branch per case."""
    seq = list(order) if order is not None else list(site.keys)

    def dispatch(a):
        for key in seq:
            a.cmp_rm_imm32('r8', key, w=False)
            a.jcc('e', 'h%d' % key)
        a.jmp('hdefault')
    return lambda: _direct_loop(site, dispatch, order=seq)


def jump_table(site, peel=()):
    """Dense indexed jump, the shape a compiler picks for a dense switch.

    peel lists keys to test with an equality compare before the table, which
    is what a profile aware compiler does for a hot case.
    """
    peel = list(peel)
    span = site.hi - site.lo
    labels = [('h%d' % v if v in site.const else 'hdefault')
              for v in range(site.lo, site.hi + 1)]

    def dispatch(a):
        for key in peel:
            a.cmp_rm_imm32('r8', key, w=False)
            a.jcc('e', 'h%d' % key)
        a.mov_rm_r('r9', 'r8', w=False)
        a.sub_rm_imm32('r9', site.lo, w=False)
        a.cmp_rm_imm32('r9', span, w=False)
        a.jcc('a', 'hdefault')
        a.jump_table(labels)
    return lambda: _direct_loop(site, dispatch)


def binary_search(site):
    """Balanced compare tree, the shape a compiler picks for a sparse switch."""
    keys = site.keys

    def emit(a, lo, hi, tag):
        if lo > hi:
            a.jmp('hdefault')
            return
        mid = (lo + hi) // 2
        k = keys[mid]
        a.cmp_rm_imm32('r8', k, w=False)
        a.jcc('e', 'h%d' % k)
        if lo == hi:
            a.jmp('hdefault')
            return
        right = 'bs%s_r' % tag
        a.jcc('a', right)
        emit(a, lo, mid - 1, tag + 'l')
        a.label(right)
        emit(a, mid + 1, hi, tag + 'r')

    def dispatch(a):
        emit(a, 0, len(keys) - 1, '0')
    return lambda: _direct_loop(site, dispatch)


# --------------------------------------------------------------------- folded

DEFAULT_BLOCK = 4096


def fold(site, block=DEFAULT_BLOCK):
    """Branchless bucketing, then one specialised run per handler.

    Per block of items:
      pass A  histogram of bucket counts, no branch on the key
      prefix  exclusive prefix sum over the bucket counts
      pass B  scatter each item index into its bucket, no branch on the key
      pass C  for each bucket, loop over its run with that handler inlined

    The key never steers a branch. The only data dependent branches left are
    the per bucket loop backs, and those are taken run_length - 1 times in a
    row, so the predictor is right on all but one iteration per bucket.
    """
    span = site.hi - site.lo
    n_slots = site.n_slots
    default_slot = n_slots - 1
    saved = ['rbx', 'rsi', 'rdi', 'r12', 'r13', 'r14', 'r15']

    def build():
        a = Asm()
        for r in saved:
            a.push(r)
        a.mov_r_rm('r14', 'r8', w=True)                      # counts base
        a.lea('r15', mem('r8', None, 1, 4 * n_slots))        # order base
        a.mov_r_rm('r12', 'rcx', w=True)                     # key cursor
        a.mov_r_rm('r13', 'rdx', w=True)                     # items remaining
        a.zero('rax')                                        # accumulator
        a.zero('rsi')                                        # global index base
        a.mov_rm_imm32('rbx', default_slot, w=False)         # cmov source

        a.label('block')
        a.test_r_rm('r13', 'r13', w=True)
        a.jcc('e', 'fold_done')
        a.mov_rm_imm32('rdx', block, w=True)
        a.cmp_rm_r('r13', 'rdx', w=True)
        a.cmov('b', 'rdx', 'r13', w=True)                    # blk = min(block, left)

        for b in range(n_slots):                             # clear counts
            a.mov_rm_imm32(mem('r14', None, 1, 4 * b), 0, w=False)

        def key_to_slot(a):
            a.mov_r_rm('r9', mem('r12', 'rcx', 4, 0), w=False)
            a.sub_rm_imm32('r9', site.lo, w=False)
            a.cmp_rm_imm32('r9', span, w=False)
            a.cmov('a', 'r9', 'rbx', w=False)

        a.zero('rcx')                                        # pass A
        a.label('pass_a')
        key_to_slot(a)
        a.inc_rm(mem('r14', 'r9', 4, 0), w=False)
        a.inc_rm('rcx', w=True)
        a.cmp_rm_r('rcx', 'rdx', w=True)
        a.jcc('b', 'pass_a')

        a.zero('r9')                                         # exclusive prefix sum
        for b in range(n_slots):
            a.mov_r_rm('r10', mem('r14', None, 1, 4 * b), w=False)
            a.mov_rm_r(mem('r14', None, 1, 4 * b), 'r9', w=False)
            a.add_r_rm('r9', 'r10', w=False)

        a.zero('rcx')                                        # pass B
        a.label('pass_b')
        key_to_slot(a)
        a.mov_r_rm('r10', mem('r14', 'r9', 4, 0), w=False)
        a.lea('r11', mem('rsi', 'rcx', 1, 0))
        a.mov_rm_r(mem('r15', 'r10', 4, 0), 'r11', w=False)
        a.inc_rm(mem('r14', 'r9', 4, 0), w=False)
        a.inc_rm('rcx', w=True)
        a.cmp_rm_r('rcx', 'rdx', w=True)
        a.jcc('b', 'pass_b')

        a.zero('r10')                                        # pass C
        for b in range(n_slots):
            key = site.lo + b if b != default_slot else None
            if key is not None and key not in site.const:
                key = None
            a.mov_r_rm('r11', mem('r14', None, 1, 4 * b), w=False)
            a.cmp_rm_r('r10', 'r11', w=False)
            a.jcc('ae', 'run%d_done' % b)
            a.label('run%d' % b)
            a.mov_r_rm('rdi', mem('r15', 'r10', 4, 0), w=False)
            _handler_body(a, site, key)
            a.add_r_rm('rax', 'r9', w=True)
            a.add_r_rm('rax', 'rdi', w=True)
            a.inc_rm('r10', w=False)
            a.cmp_rm_r('r10', 'r11', w=False)
            a.jcc('b', 'run%d' % b)
            a.label('run%d_done' % b)

        a.add_r_rm('rsi', 'rdx', w=True)                     # next block
        a.lea('r12', mem('r12', 'rdx', 4, 0))
        a.sub_r_rm('r13', 'rdx', w=True)
        a.jmp('block')

        a.label('fold_done')
        for r in reversed(saved):
            a.pop(r)
        a.ret()
        return a

    return build


def scratch_words(site, block=DEFAULT_BLOCK):
    return site.n_slots + block + 16

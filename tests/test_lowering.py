"""Every lowering must return exactly the same accumulator.

This is the correctness argument for the folded kernel: it runs the handlers
in a different order than the source program would, so the only thing that
makes it a legal rewrite is that the per item results are combined with an
operation that does not care about order. These tests pin that down on
awkward shapes (sparse keys, keys outside the domain, extra handler work).
"""
import random
import unittest

from dispatchfold import lowering as L
from dispatchfold.jit import Buffer, Kernel, time_kernel


def check(site, stream, block=256):
    ref = site.reference(stream)
    buf = Buffer(stream, L.scratch_words(site, block))
    results = {}
    variants = [('jump-table', L.jump_table(site)),
                ('binary-search', L.binary_search(site)),
                ('fold', L.fold(site, block))]
    if len(site.keys) <= 24:
        variants.append(('if-chain', L.chain(site)))
        variants.append(('peeled', L.jump_table(site, peel=site.keys[:2])))
    for name, builder in variants:
        kern = Kernel(name, builder)
        try:
            results[name] = kern(buf.keys_ptr, buf.n, buf.scratch_ptr)
        finally:
            kern.free()
    return ref, results


class TestLowerings(unittest.TestCase):
    def assert_all_agree(self, site, stream):
        ref, results = check(site, stream)
        for name, got in results.items():
            self.assertEqual(got, ref, '%s returned %d, reference %d'
                             % (name, got, ref))

    def test_dense_small(self):
        rng = random.Random(1)
        site = L.Site(range(8), 0, 7)
        self.assert_all_agree(site, [rng.randrange(8) for _ in range(5000)])

    def test_sparse_keys_with_gaps(self):
        rng = random.Random(2)
        keys = [3, 9, 10, 11, 40, 41, 97]
        site = L.Site(keys, 0, 100)
        stream = [rng.choice(keys) for _ in range(5000)]
        self.assert_all_agree(site, stream)

    def test_keys_outside_the_domain_hit_the_default(self):
        rng = random.Random(3)
        keys = [5, 6, 7]
        site = L.Site(keys, 5, 7)
        stream = [rng.choice([5, 6, 7, 0, 99, 4000, 0xFFFFFFFF])
                  for _ in range(5000)]
        self.assert_all_agree(site, stream)

    def test_handler_work_is_identical_across_lowerings(self):
        rng = random.Random(4)
        for ops in (1, 5):
            site = L.Site(range(12), 0, 11, body_ops=ops)
            self.assert_all_agree(site, [rng.randrange(12) for _ in range(4000)])

    def test_stream_shorter_than_one_block(self):
        site = L.Site(range(4), 0, 3)
        self.assert_all_agree(site, [1, 2, 3, 0, 3, 3, 1])

    def test_stream_not_a_multiple_of_the_block(self):
        rng = random.Random(5)
        site = L.Site(range(6), 0, 5)
        stream = [rng.randrange(6) for _ in range(256 * 3 + 7)]
        self.assert_all_agree(site, stream)

    def test_empty_stream(self):
        site = L.Site(range(4), 0, 3)
        ref, results = check(site, [])
        self.assertEqual(ref, 0)
        for name, got in results.items():
            self.assertEqual(got, 0, name)

    def test_large_index_term_does_not_overflow_differently(self):
        rng = random.Random(6)
        site = L.Site(range(32), 0, 31)
        self.assert_all_agree(site, [rng.randrange(32) for _ in range(70000)])

    def test_timing_harness_reports_a_positive_rate(self):
        site = L.Site(range(4), 0, 3)
        stream = [0, 1, 2, 3] * 1000
        buf = Buffer(stream, L.scratch_words(site, 256))
        kern = Kernel('t', L.jump_table(site))
        try:
            ns, acc = time_kernel(kern, buf, reps=3, warmup=1)
        finally:
            kern.free()
        self.assertGreater(ns, 0.0)
        self.assertEqual(acc, site.reference(stream))


if __name__ == '__main__':
    unittest.main()

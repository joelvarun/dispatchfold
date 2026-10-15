"""Load emitted machine code into executable memory and time it.

Every dispatch kernel in this package has the same C signature:

    uint64_t kernel(const uint32_t *keys, uint64_t n, uint32_t *scratch);

and the same semantics: for every key in keys[0:n] it selects a handler by
the key value and accumulates handler_constant + item_index. Scratch is only
touched by the folded kernel. Because the accumulator is a plain sum, the
result does not depend on the order the items are handled in, which is what
lets the folded kernel reorder work and still be checked against the direct
kernels bit for bit.
"""
import ctypes
import platform
import time

MEM_COMMIT_RESERVE = 0x3000
PAGE_EXECUTE_READWRITE = 0x40
PAGE_READWRITE = 0x04
MEM_RELEASE = 0x8000

KERNEL_TYPE = ctypes.CFUNCTYPE(
    ctypes.c_uint64, ctypes.c_void_p, ctypes.c_uint64, ctypes.c_void_p)


def _k32():
    if platform.system() != 'Windows':
        raise RuntimeError(
            'dispatchfold needs Windows x86-64: it allocates RWX pages with '
            'VirtualAlloc and emits x86-64 directly.')
    k = ctypes.windll.kernel32
    k.VirtualAlloc.restype = ctypes.c_void_p
    k.VirtualAlloc.argtypes = [ctypes.c_void_p, ctypes.c_size_t,
                               ctypes.c_uint32, ctypes.c_uint32]
    k.VirtualFree.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_uint32]
    return k


class Kernel(object):
    """An assembled dispatch loop living in RWX memory."""

    def __init__(self, name, asm_builder, slack=1 << 16):
        self.name = name
        k = _k32()
        # Pass 1: assemble at a dummy base to learn the exact size, then
        # allocate and assemble again at the real address. Absolute jump-table
        # entries need the final address, and the size never depends on it.
        probe = asm_builder()
        size = len(probe.link(0)) + slack
        size = (size + 0xFFF) & ~0xFFF
        self._size = size
        self.addr = k.VirtualAlloc(None, size, MEM_COMMIT_RESERVE,
                                   PAGE_EXECUTE_READWRITE)
        if not self.addr:
            raise MemoryError('VirtualAlloc failed for %d bytes' % size)
        code = asm_builder().link(self.addr)
        if len(code) > size:
            raise RuntimeError('code grew past the reserved page')
        ctypes.memmove(self.addr, code, len(code))
        self.code_size = len(code)
        self.fn = KERNEL_TYPE(self.addr)

    def __call__(self, keys, n, scratch):
        return self.fn(keys, n, scratch)

    def free(self):
        if self.addr:
            _k32().VirtualFree(ctypes.c_void_p(self.addr), 0, MEM_RELEASE)
            self.addr = None


class Buffer(object):
    """A uint32 array plus a scratch area, both page aligned by VirtualAlloc."""

    def __init__(self, values, scratch_words):
        self.n = len(values)
        self.keys = (ctypes.c_uint32 * max(1, self.n))(*values)
        self.scratch = (ctypes.c_uint32 * max(1, scratch_words))()

    @property
    def keys_ptr(self):
        return ctypes.cast(self.keys, ctypes.c_void_p)

    @property
    def scratch_ptr(self):
        return ctypes.cast(self.scratch, ctypes.c_void_p)


def time_kernel(kernel, buf, reps=25, warmup=3):
    """Return (best nanoseconds per item, returned accumulator).

    The minimum over repetitions is used rather than the mean: the kernel is
    deterministic, so every sample above the minimum is interference from the
    rest of the machine.
    """
    for _ in range(warmup):
        kernel(buf.keys_ptr, buf.n, buf.scratch_ptr)
    best = float('inf')
    acc = None
    for _ in range(reps):
        t0 = time.perf_counter_ns()
        acc = kernel(buf.keys_ptr, buf.n, buf.scratch_ptr)
        dt = time.perf_counter_ns() - t0
        if dt < best:
            best = dt
    return best / buf.n, acc

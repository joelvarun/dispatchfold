"""Minimal x86-64 assembler: just enough to emit dispatch loops at runtime.

Only a handful of instructions are encoded, and only the Windows x64 volatile
registers plus rbx are used by the code generators in this package. Every
encoding is covered by tests/test_x64.py against bytes verified by execution.
"""
import struct

REGS = {
    'rax': 0, 'rcx': 1, 'rdx': 2, 'rbx': 3, 'rsp': 4, 'rbp': 5, 'rsi': 6,
    'rdi': 7, 'r8': 8, 'r9': 9, 'r10': 10, 'r11': 11, 'r12': 12, 'r13': 13,
    'r14': 14, 'r15': 15,
}

CC = {'e': 4, 'ne': 5, 'b': 2, 'ae': 3, 'be': 6, 'a': 7,
      'l': 12, 'ge': 13, 'le': 14, 'g': 15}


def _r(name):
    try:
        return REGS[name]
    except KeyError:
        raise ValueError('unknown register %r' % (name,))


class Mem(object):
    """[base + index*scale + disp]"""
    __slots__ = ('base', 'index', 'scale', 'disp')

    def __init__(self, base, index=None, scale=1, disp=0):
        if scale not in (1, 2, 4, 8):
            raise ValueError('bad scale %r' % (scale,))
        self.base, self.index, self.scale, self.disp = base, index, scale, disp


def mem(base, index=None, scale=1, disp=0):
    return Mem(base, index, scale, disp)


class Asm(object):
    def __init__(self):
        self.buf = bytearray()
        self.labels = {}
        self._rel = []
        self._tables = []

    def label(self, name):
        if name in self.labels:
            raise ValueError('duplicate label %r' % (name,))
        self.labels[name] = len(self.buf)

    def _emit(self, *bs):
        self.buf.extend(bs)

    def _i32(self, v):
        self.buf.extend(struct.pack('<i', v))

    def _u32(self, v):
        self.buf.extend(struct.pack('<I', v & 0xFFFFFFFF))

    def _opcode(self, opcode):
        if isinstance(opcode, tuple):
            self._emit(*opcode)
        else:
            self._emit(opcode)

    def _enc(self, opcode, reg, rm, w, ext=None):
        rfield = ext if ext is not None else _r(reg)
        if isinstance(rm, Mem):
            base = _r(rm.base)
            index = None if rm.index is None else _r(rm.index)
            rex = 0x40 | (0x08 if w else 0) | ((rfield >> 3) << 2) | (base >> 3)
            if index is not None:
                rex |= (index >> 3) << 1
            if rex != 0x40:
                self._emit(rex)
            self._opcode(opcode)
            modrm = 0x80 | ((rfield & 7) << 3) | 4     # mod=10 (disp32), rm=SIB
            if index is None:
                sib = (4 << 3) | (base & 7)            # index=100 means none
            else:
                sib = ({1: 0, 2: 1, 4: 2, 8: 3}[rm.scale] << 6) \
                    | ((index & 7) << 3) | (base & 7)
            self._emit(modrm, sib)
            self._i32(rm.disp)
        else:
            b = _r(rm)
            rex = 0x40 | (0x08 if w else 0) | ((rfield >> 3) << 2) | (b >> 3)
            if rex != 0x40:
                self._emit(rex)
            self._opcode(opcode)
            self._emit(0xC0 | ((rfield & 7) << 3) | (b & 7))

    def mov_r_rm(self, dst, src, w=True):
        self._enc(0x8B, dst, src, w)

    def mov_rm_r(self, dst, src, w=True):
        self._enc(0x89, src, dst, w)

    def mov_rm_imm32(self, dst, imm, w=False):
        self._enc(0xC7, None, dst, w, ext=0)
        self._u32(imm)

    def movabs(self, dst, imm64):
        d = _r(dst)
        self._emit(0x48 | (d >> 3), 0xB8 + (d & 7))
        self.buf.extend(struct.pack('<Q', imm64))

    def lea(self, dst, m):
        self._enc(0x8D, dst, m, True)

    def add_r_rm(self, dst, src, w=True):
        self._enc(0x03, dst, src, w)

    def add_rm_imm32(self, dst, imm, w=False):
        self._enc(0x81, None, dst, w, ext=0)
        self._u32(imm)

    def sub_rm_imm32(self, dst, imm, w=False):
        self._enc(0x81, None, dst, w, ext=5)
        self._u32(imm)

    def xor_rm_imm32(self, dst, imm, w=False):
        self._enc(0x81, None, dst, w, ext=6)
        self._u32(imm)

    def cmp_rm_imm32(self, dst, imm, w=False):
        self._enc(0x81, None, dst, w, ext=7)
        self._u32(imm)

    def cmp_rm_r(self, a, b, w=True):
        self._enc(0x39, b, a, w)

    def inc_rm(self, dst, w=False):
        self._enc(0xFF, None, dst, w, ext=0)

    def xor_r_rm(self, dst, src, w=False):
        self._enc(0x33, dst, src, w)

    def zero(self, dst):
        self.xor_r_rm(dst, dst, w=False)

    def test_r_rm(self, a, b, w=True):
        self._enc(0x85, b, a, w)

    def imul_r_rm(self, dst, src, w=True):
        self._enc((0x0F, 0xAF), dst, src, w)

    def sub_r_rm(self, dst, src, w=True):
        self._enc(0x2B, dst, src, w)

    def cmov(self, cc, dst, src, w=False):
        self._enc((0x0F, 0x40 | CC[cc]), dst, src, w)

    def jmp(self, label):
        self._emit(0xE9)
        self._rel.append((len(self.buf), label))
        self._u32(0)

    def jcc(self, cc, label):
        self._emit(0x0F, 0x80 | CC[cc])
        self._rel.append((len(self.buf), label))
        self._u32(0)

    def jmp_r(self, reg):
        self._enc(0xFF, None, reg, False, ext=4)

    def jump_table(self, labels, scratch='r10', index='r9', target='r11'):
        d = _r(scratch)
        self._emit(0x48 | (d >> 3), 0xB8 + (d & 7))
        self._tables.append((len(self.buf), list(labels)))
        self.buf.extend(b'\x00' * 8)
        self.mov_r_rm(target, mem(scratch, index, 8, 0))
        self.jmp_r(target)

    def push(self, reg):
        d = _r(reg)
        if d >> 3:
            self._emit(0x41)
        self._emit(0x50 + (d & 7))

    def pop(self, reg):
        d = _r(reg)
        if d >> 3:
            self._emit(0x41)
        self._emit(0x58 + (d & 7))

    def ret(self):
        self._emit(0xC3)

    def link(self, base_address):
        """Resolve labels and jump tables for code loaded at base_address."""
        while len(self.buf) % 8:
            self.buf.append(0x90)
        for imm_off, labels in self._tables:
            struct.pack_into('<Q', self.buf, imm_off, base_address + len(self.buf))
            for name in labels:
                self.buf.extend(struct.pack('<Q', base_address + self.labels[name]))
        for off, name in self._rel:
            if name not in self.labels:
                raise ValueError('unresolved label %r' % (name,))
            struct.pack_into('<i', self.buf, off, self.labels[name] - (off + 4))
        return bytes(self.buf)

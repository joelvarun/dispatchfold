"""Encoding tests for the assembler.

The expected bytes here were produced by hand from the Intel encoding rules
and then confirmed by executing the kernels that use them: a wrong encoding
either faults or returns the wrong accumulator, and test_lowering.py checks
the accumulator.
"""
import unittest

from dispatchfold.x64 import Asm, mem


def enc(fn, *args, **kwargs):
    a = Asm()
    fn(a, *args, **kwargs)
    return bytes(a.buf)


class TestEncoding(unittest.TestCase):
    def test_reg_to_reg(self):
        self.assertEqual(enc(Asm.mov_rm_r, 'r9', 'r8', w=False),
                         bytes([0x45, 0x89, 0xC1]))
        self.assertEqual(enc(Asm.add_r_rm, 'rax', 'r9', w=True),
                         bytes([0x49, 0x03, 0xC1]))
        self.assertEqual(enc(Asm.cmp_rm_r, 'rbx', 'rdx', w=True),
                         bytes([0x48, 0x39, 0xD3]))

    def test_immediates(self):
        self.assertEqual(enc(Asm.mov_rm_imm32, 'r9', 0x11223344, w=False),
                         bytes([0x41, 0xC7, 0xC1, 0x44, 0x33, 0x22, 0x11]))
        self.assertEqual(enc(Asm.sub_rm_imm32, 'r9', 1, w=False),
                         bytes([0x41, 0x81, 0xE9, 1, 0, 0, 0]))
        self.assertEqual(enc(Asm.cmp_rm_imm32, 'r8', 7, w=False),
                         bytes([0x41, 0x81, 0xF8, 7, 0, 0, 0]))

    def test_memory_operands(self):
        # mov r8d, [rcx + rbx*4]
        self.assertEqual(enc(Asm.mov_r_rm, 'r8', mem('rcx', 'rbx', 4, 0),
                             w=False),
                         bytes([0x44, 0x8B, 0x84, 0x99, 0, 0, 0, 0]))
        # inc dword [r14 + r9*4]
        self.assertEqual(enc(Asm.inc_rm, mem('r14', 'r9', 4, 0), w=False),
                         bytes([0x43, 0xFF, 0x84, 0x8E, 0, 0, 0, 0]))
        # mov r11d, [r14 + 0x40]
        self.assertEqual(enc(Asm.mov_r_rm, 'r11', mem('r14', None, 1, 0x40),
                             w=False),
                         bytes([0x45, 0x8B, 0x9C, 0x26, 0x40, 0, 0, 0]))

    def test_cmov_and_control(self):
        self.assertEqual(enc(Asm.cmov, 'a', 'r9', 'rbx', w=False),
                         bytes([0x44, 0x0F, 0x47, 0xCB]))
        self.assertEqual(enc(Asm.jmp_r, 'r11'), bytes([0x41, 0xFF, 0xE3]))
        self.assertEqual(enc(Asm.ret), bytes([0xC3]))
        self.assertEqual(enc(Asm.push, 'r12'), bytes([0x41, 0x54]))
        self.assertEqual(enc(Asm.pop, 'rbx'), bytes([0x5B]))

    def test_relative_fixups(self):
        a = Asm()
        a.label('top')
        a.ret()
        a.jmp('top')
        code = a.link(0)
        # the jmp starts at offset 1 and ends at offset 6, and the target
        # is offset 0, so the stored displacement must be -6
        self.assertEqual(code[0:2], bytes([0xC3, 0xE9]))
        self.assertEqual(code[2:6], (-6).to_bytes(4, 'little', signed=True))

    def test_jump_table_holds_absolute_addresses(self):
        a = Asm()
        a.label('a')
        a.ret()
        a.jump_table(['a', 'a'])
        code = a.link(0x1000)
        self.assertIn((0x1000).to_bytes(8, 'little'), code)

    def test_unresolved_label_is_an_error(self):
        a = Asm()
        a.jmp('nowhere')
        self.assertRaises(ValueError, a.link, 0)


if __name__ == '__main__':
    unittest.main()

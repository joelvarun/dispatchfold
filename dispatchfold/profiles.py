"""Real dispatch key streams harvested from the local machine.

Nothing here is synthetic. Each workload is the actual sequence of keys that
some real program's switch would see, in the order it would see them, because
order is what decides whether the branch predictor can keep up.
"""
import ast
import os
import sys

MIN_ITEMS = 1 << 16


class Workload(object):
    def __init__(self, name, stream, keys, lo, hi, what, source):
        self.name = name
        self.stream = stream
        self.keys = sorted(set(keys))
        self.lo, self.hi = lo, hi
        self.what = what          # what the dispatch would be doing
        self.source = source      # where the keys came from

    def __repr__(self):
        return '<Workload %s n=%d cases=%d>' % (
            self.name, len(self.stream), len(self.keys))


def _stdlib_dir():
    return os.path.dirname(os.__file__)


def _read_stdlib_text(budget=1 << 21):
    chunks, total = [], 0
    root = _stdlib_dir()
    for name in sorted(os.listdir(root)):
        if not name.endswith('.py'):
            continue
        try:
            with open(os.path.join(root, name), 'rb') as fh:
                data = fh.read()
        except OSError:
            continue
        chunks.append(data)
        total += len(data)
        if total >= budget:
            break
    return b''.join(chunks)


# ------------------------------------------------------- 1. interpreter loop

def _trace_workload():
    """A small but real mix: dict and list churn, string work, sorting."""
    import json
    rows = [{'k': i, 'v': 'item-%d' % i} for i in range(120)]
    back = json.loads(json.dumps(rows))
    names = sorted((r['v'] for r in back), key=lambda s: (len(s), s))
    acc = 0
    for i, s in enumerate(names):
        acc += sum(ord(c) for c in s) * (i % 7)
        if acc % 3 == 0:
            acc //= 2
    return acc


def cpython_opcodes(limit=200000):
    """The dynamic opcode stream of the CPython eval loop, by opcode tracing.

    This is the canonical switch: one case per bytecode, taken once per
    executed instruction. The trace comes from sys.settrace opcode events, so
    it is real execution order rather than a static walk over code objects.
    """
    import dis
    stream = []
    append = stream.append

    def tracer(frame, event, arg):
        if event == 'call':
            frame.f_trace_opcodes = True
            return tracer
        if event == 'opcode':
            if len(stream) >= limit:
                sys.settrace(None)
                return None
            append(frame.f_code.co_code[frame.f_lasti])
        return tracer

    sys.settrace(tracer)
    try:
        while len(stream) < limit:
            _trace_workload()
    finally:
        sys.settrace(None)
    valid = [i for i, nm in enumerate(dis.opname) if not nm.startswith('<')]
    return Workload('cpython-opcodes', stream[:limit], valid, 0, 255,
                    'bytecode dispatch in an interpreter eval loop',
                    'sys.settrace opcode events over a json plus sort workload')


# ---------------------------------------------------------------- 2. ast kinds

def ast_nodes(budget=1 << 20):
    """Node kind dispatch in a compiler or linter visitor, in DFS order."""
    order = []
    kinds = {}

    def visit(node):
        name = type(node).__name__
        idx = kinds.get(name)
        if idx is None:
            idx = kinds[name] = len(kinds)
        order.append(idx)
        for child in ast.iter_child_nodes(node):
            visit(child)

    total = 0
    root = _stdlib_dir()
    for name in sorted(os.listdir(root)):
        if not name.endswith('.py'):
            continue
        try:
            with open(os.path.join(root, name), 'rb') as fh:
                src = fh.read()
            tree = ast.parse(src)
        except (OSError, SyntaxError, ValueError, RecursionError):
            continue
        visit(tree)
        total += len(src)
        if total >= budget and len(order) >= MIN_ITEMS:
            break
    return Workload('ast-kinds', order, range(len(kinds)), 0,
                    max(1, len(kinds) - 1),
                    'AST node kind dispatch in a visitor',
                    '%d node kinds over the Python stdlib sources' % len(kinds))


# --------------------------------------------------------------- 3. text bytes

def text_bytes():
    """Byte dispatch in a UTF-8 scanner or lexer over real source text."""
    data = _read_stdlib_text()
    return Workload('text-bytes', list(data), range(256), 0, 255,
                    'byte dispatch in a UTF-8 scanner or lexer',
                    'first %d bytes of the Python stdlib sources' % len(data))


# ---------------------------------------------------------- 4. compressed data

def compressed_bytes(budget=1 << 20):
    """Byte dispatch over a real compressed stream: near maximum entropy."""
    root = _stdlib_dir()
    found = None
    for dirpath, _dirs, files in os.walk(os.path.join(root, 'ensurepip')):
        for f in sorted(files):
            if f.endswith('.whl'):
                found = os.path.join(dirpath, f)
                break
        if found:
            break
    if found is None:
        raise RuntimeError('no bundled wheel found to read compressed bytes from')
    with open(found, 'rb') as fh:
        data = fh.read(budget)
    return Workload('compressed-bytes', list(data), range(256), 0, 255,
                    'byte dispatch in an entropy or container decoder',
                    'first %d bytes of %s' % (len(data), os.path.basename(found)))


# ------------------------------------------------------------- 5. machine code

def pe_code_bytes(budget=1 << 20):
    """Opcode byte dispatch in an instruction decoder over real machine code."""
    root = os.environ.get('SystemRoot', 'C:/Windows')
    path = os.path.join(root, 'System32', 'shell32.dll')
    with open(path, 'rb') as fh:
        fh.seek(1 << 16)
        data = fh.read(budget)
    return Workload('pe-code-bytes', list(data), range(256), 0, 255,
                    'opcode byte dispatch in an instruction decoder',
                    '%d bytes from the body of %s'
                    % (len(data), os.path.basename(path)))


# -------------------------------------------------------------- 6. json lexer

JSON_CLASSES = ['space', 'newline', 'digit', 'quote', 'escape', 'lbrace',
                'rbrace', 'lbracket', 'rbracket', 'colon', 'comma', 'letter',
                'minus', 'other']

_PUNCT = {123: 5, 125: 6, 91: 7, 93: 8, 58: 9, 44: 10, 45: 12}


def _json_class(byte):
    if byte in (32, 9, 13):
        return 0
    if byte == 10:
        return 1
    if 48 <= byte <= 57:
        return 2
    if byte == 34:
        return 3
    if byte == 92:
        return 4
    hit = _PUNCT.get(byte)
    if hit is not None:
        return hit
    if 65 <= byte <= 90 or 97 <= byte <= 122:
        return 11
    return 13


def json_classes(budget=1 << 20):
    """Character class dispatch in a JSON scanner over a real JSON file."""
    path = os.path.join(_stdlib_dir(), 'test', 'levenshtein_examples.json')
    if not os.path.exists(path):
        raise RuntimeError('expected stdlib test data at %s' % path)
    with open(path, 'rb') as fh:
        data = fh.read(budget)
    return Workload('json-classes', [_json_class(b) for b in data],
                    range(len(JSON_CLASSES)), 0, len(JSON_CLASSES) - 1,
                    'character class dispatch in a JSON scanner',
                    '%d bytes of %s' % (len(data), os.path.basename(path)))


ALL = [cpython_opcodes, ast_nodes, text_bytes, compressed_bytes,
       pe_code_bytes, json_classes]


def load_all(verbose=True):
    out = []
    for fn in ALL:
        w = fn()
        if len(w.stream) < MIN_ITEMS:
            raise RuntimeError('workload %s only produced %d items'
                               % (w.name, len(w.stream)))
        if verbose:
            print('  %-17s %7d keys %4d cases  %s'
                  % (w.name, len(w.stream), len(w.keys), w.source))
        out.append(w)
    return out

#!/usr/bin/env python3
"""GGUF headers with the standard library only (no numpy, no gguf-py): metadata, tensor names, types, shapes and
absolute data offsets, for one file or every shard of a split model.

Only the header is read (mapped, not copied), so a shard of tens of GB costs its metadata.  Arrays are decoded too
(a tokenizer's vocabulary takes a fraction of a second).
usage: gguf_header.py FILE [--tensors]   -> the metadata of FILE (one line per key; arrays as their length), and with
                                            --tensors one line per tensor: name type shape offset nbytes
"""
import glob
import mmap
import re
import struct
import sys
from typing import NamedTuple

# ggml_type id -> (name, elements per block, bytes per block), as ggml/include/ggml.h and ggml.c define them
TYPES = {
    0: ('F32', 1, 4), 1: ('F16', 1, 2), 2: ('Q4_0', 32, 18), 3: ('Q4_1', 32, 20), 6: ('Q5_0', 32, 22),
    7: ('Q5_1', 32, 24), 8: ('Q8_0', 32, 34), 9: ('Q8_1', 32, 40), 10: ('Q2_K', 256, 84), 11: ('Q3_K', 256, 110),
    12: ('Q4_K', 256, 144), 13: ('Q5_K', 256, 176), 14: ('Q6_K', 256, 210), 15: ('Q8_K', 256, 292),
    16: ('IQ2_XXS', 256, 66), 17: ('IQ2_XS', 256, 74), 18: ('IQ3_XXS', 256, 98), 19: ('IQ1_S', 256, 50),
    20: ('IQ4_NL', 32, 18), 21: ('IQ3_S', 256, 110), 22: ('IQ2_S', 256, 82), 23: ('IQ4_XS', 256, 136),
    24: ('I8', 1, 1), 25: ('I16', 1, 2), 26: ('I32', 1, 4), 27: ('I64', 1, 8), 28: ('F64', 1, 8),
    29: ('IQ1_M', 256, 56), 30: ('BF16', 1, 2), 34: ('TQ1_0', 256, 54), 35: ('TQ2_0', 256, 66),
    39: ('MXFP4', 32, 17), 40: ('NVFP4', 64, 36), 41: ('Q1_0', 128, 18), 42: ('Q2_0', 64, 18),
}
# metadata value types: id -> struct format (8 = string, 9 = array)
SCALARS = {0: 'B', 1: 'b', 2: 'H', 3: 'h', 4: 'I', 5: 'i', 6: 'f', 7: '?', 10: 'Q', 11: 'q', 12: 'd'}
SHARD = re.compile(r'(.*)-\d{5}-of-(\d{5})\.gguf$')


class Tensor(NamedTuple):
    name: str
    type: int       # ggml_type id
    shape: tuple    # ne[0], ne[1], ... (ggml order: ne[0] is the row length)
    offset: int     # absolute offset of the data in the file
    nbytes: int
    path: str

    @property
    def type_name(self):
        return TYPES[self.type][0] if self.type in TYPES else f'type{self.type}'


class Header(NamedTuple):
    path: str
    version: int
    kv: dict
    tensors: list
    alignment: int
    data_offset: int


class _Cursor:
    def __init__(self, buf):
        self.buf, self.pos = buf, 0

    def take(self, fmt):
        v = struct.unpack_from('<' + fmt, self.buf, self.pos)
        self.pos += struct.calcsize('<' + fmt)
        return v[0] if len(v) == 1 else v

    def string(self):
        n = self.take('Q')
        s = self.buf[self.pos:self.pos + n]
        self.pos += n
        return bytes(s).decode('utf-8', errors='replace')

    def value(self, vtype):
        if vtype in SCALARS:
            return self.take(SCALARS[vtype])
        if vtype == 8:
            return self.string()
        if vtype == 9:
            etype, n = self.take('I'), self.take('Q')
            if etype in SCALARS:
                fmt = SCALARS[etype]
                v = struct.unpack_from(f'<{n}{fmt}', self.buf, self.pos)
                self.pos += n * struct.calcsize(fmt)
                return list(v)
            return [self.value(etype) for _ in range(n)]
        raise ValueError(f'unknown GGUF metadata type {vtype} at byte {self.pos}')


def read(path):
    """The header of one GGUF file."""
    with open(path, 'rb') as f, mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ) as m:
        c = _Cursor(m)
        magic, version = c.take('4s'), c.take('I')
        if magic != b'GGUF' or version < 2:
            raise ValueError(f'{path}: not a GGUF file of version 2 or later')
        n_tensors, n_kv = c.take('Q'), c.take('Q')
        kv = {}
        for _ in range(n_kv):
            key = c.string()
            kv[key] = c.value(c.take('I'))
        infos = []
        for _ in range(n_tensors):
            name = c.string()
            ndim = c.take('I')
            shape = tuple(c.take(f'{ndim}Q')) if ndim > 1 else (c.take('Q'),)
            infos.append((name, shape, c.take('I'), c.take('Q')))
        alignment = int(kv.get('general.alignment', 32))
        data_offset = (c.pos + alignment - 1) // alignment * alignment
    tensors = []
    for name, shape, t, off in infos:
        n = 1
        for d in shape:
            n *= d
        if t not in TYPES:
            raise ValueError(f'{path}: tensor {name} has ggml type {t}, which this reader does not know')
        _, block, size = TYPES[t]
        tensors.append(Tensor(name, t, shape, data_offset + off, n * size // block, path))
    return Header(path, version, kv, tensors, alignment, data_offset)


def shards(path):
    """Every shard of the split model whose shard PATH is (sorted), or [PATH] for a single file."""
    m = SHARD.match(path)
    return sorted(glob.glob(f'{glob.escape(m.group(1))}-*-of-{m.group(2)}.gguf')) if m else [path]


def read_model(path):
    """The headers of every shard of the model PATH belongs to."""
    return [read(p) for p in shards(path)]


if __name__ == '__main__':
    a = sys.argv[1:]
    if len(a) not in (1, 2) or (len(a) == 2 and a[1] != '--tensors'):
        sys.exit(__doc__)
    h = read(a[0])
    print(f'version {h.version} alignment {h.alignment} data_offset {h.data_offset} tensors {len(h.tensors)}')
    for k, v in h.kv.items():
        print(k, f'[{len(v)}]' if isinstance(v, list) else repr(v))
    if len(a) == 2:
        for t in h.tensors:
            print(t.name, t.type_name, 'x'.join(map(str, t.shape)), t.offset, t.nbytes)

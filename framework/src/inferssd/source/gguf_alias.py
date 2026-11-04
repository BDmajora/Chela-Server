"""Strict, standard-library GGUF alias index of the routed experts the SSD executor reads.

The legacy metadata reader intentionally retains historical observation semantics.
This separate reader checks the complete header and every tensor extent without
mapping or reading tensor payloads. Only little-endian GGUF versions 2 and 3 are
supported.
"""

from __future__ import annotations

import argparse
import codecs
import csv
import glob
import json
import os
import re
import stat
import struct
import subprocess
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, BinaryIO

from inferssd.platform.captured_command import run_captured
from inferssd.source.header import SCALARS

ALIGNMENT = 4096
MAX_I64 = (1 << 63) - 1
# Binary reads on Windows, where os.open otherwise translates text; on POSIX a
# FIFO must not block the open before the regular-file check rejects it.
OPEN_FLAGS = os.O_RDONLY | getattr(os, "O_BINARY", 0) | getattr(os, "O_NONBLOCK", 0)
# Native ggml.c type_traits / ggml-common.h geometry. Keep this independent of
# the original metadata table (in particular, its Q8_1 size is different).
# The repository native-contract gate checks every admitted type against the
# compiled enum, block expression and sizeof expression from type_traits.
NATIVE_TYPES = {
    0: ("F32", 1, 4),
    1: ("F16", 1, 2),
    2: ("Q4_0", 32, 18),
    3: ("Q4_1", 32, 20),
    6: ("Q5_0", 32, 22),
    7: ("Q5_1", 32, 24),
    8: ("Q8_0", 32, 34),
    9: ("Q8_1", 32, 36),
    10: ("Q2_K", 256, 84),
    11: ("Q3_K", 256, 110),
    12: ("Q4_K", 256, 144),
    13: ("Q5_K", 256, 176),
    14: ("Q6_K", 256, 210),
    15: ("Q8_K", 256, 292),
    16: ("IQ2_XXS", 256, 66),
    17: ("IQ2_XS", 256, 74),
    18: ("IQ3_XXS", 256, 98),
    19: ("IQ1_S", 256, 50),
    20: ("IQ4_NL", 32, 18),
    21: ("IQ3_S", 256, 110),
    22: ("IQ2_S", 256, 82),
    23: ("IQ4_XS", 256, 136),
    24: ("I8", 1, 1),
    25: ("I16", 1, 2),
    26: ("I32", 1, 4),
    27: ("I64", 1, 8),
    28: ("F64", 1, 8),
    29: ("IQ1_M", 256, 56),
    30: ("BF16", 1, 2),
    34: ("TQ1_0", 256, 54),
    35: ("TQ2_0", 256, 66),
    39: ("MXFP4", 32, 17),
    40: ("NVFP4", 64, 36),
    41: ("Q1_0", 128, 18),
    42: ("Q2_0", 64, 18),
}
EXPERT_RE = re.compile(r"^blk\.(?P<layer>\d+)\.ffn_(?P<kind>up|gate|down)_exps\.weight$")
SHARD_RE = re.compile(r"(.+)-(\d{5})-of-(\d{5})\.gguf$")
FIELDS = (
    "tensor",
    "expert_idx",
    "nbytes",
    "source_path",
    "offset",
    "kind",
    "layer",
    "type",
    "aligned",
    "shard_index",
)


class _Reader:
    def __init__(self, stream: BinaryIO, length: int) -> None:
        self.stream, self.length, self.offset = stream, length, 0

    def span(self, count: int) -> None:
        if count < 0 or count > self.length - self.offset:
            raise ValueError(f"truncated GGUF header at byte {self.offset}: need {count} bytes")

    def read(self, count: int) -> bytes:
        self.span(count)
        pieces = []
        remaining = count
        while remaining:
            chunk = self.stream.read(min(remaining, 1 << 20))
            if not chunk:
                raise ValueError(f"short GGUF header read at byte {self.offset}")
            pieces.append(chunk)
            self.offset += len(chunk)
            remaining -= len(chunk)
        return b"".join(pieces)

    def take(self, fmt: str) -> Any:
        values = struct.unpack("<" + fmt, self.read(struct.calcsize("<" + fmt)))
        return values[0] if len(values) == 1 else values

    def skip(self, count: int) -> None:
        self.span(count)
        self.stream.seek(count, os.SEEK_CUR)
        self.offset += count

    def string(self, *, capture: bool = True, max_bytes: int | None = None) -> str:
        length = int(self.take("Q"))
        if max_bytes is not None and length > max_bytes:
            raise RuntimeError(f"tensor name too long for runtime key: {length} UTF-8 bytes")
        self.span(length)
        decoder = codecs.getincrementaldecoder("utf-8")("strict")
        pieces = []
        while length:
            count = min(length, 1 << 20)
            text = decoder.decode(self.read(count), final=count == length)
            if capture:
                pieces.append(text)
            length -= count
        # Finalize even an empty string, without accepting incomplete UTF-8.
        tail = decoder.decode(b"", final=True)
        if capture:
            pieces.append(tail)
        return "".join(pieces)

    def value(self, kind: int, *, capture: bool = False) -> Any:
        if kind in SCALARS:
            return self.take(SCALARS[kind])
        if kind == 8:
            return self.string(capture=capture)
        if kind == 9:
            element, count = int(self.take("I")), int(self.take("Q"))
            if element in SCALARS:
                self.skip(count * struct.calcsize("<" + SCALARS[element]))
            elif element == 8:
                if count > (self.length - self.offset) // 8:
                    raise ValueError("truncated GGUF string array")
                for _ in range(count):
                    self.string(capture=False)
            else:
                raise ValueError(f"unsupported GGUF array element type {element}")
            return None
        raise ValueError(f"unknown GGUF metadata type {kind} at byte {self.offset}")


@dataclass(frozen=True)
class Tensor:
    name: str
    shape: tuple[int, ...]
    type_id: int
    offset: int
    nbytes: int


@dataclass(frozen=True)
class Header:
    path: Path
    tensors: tuple[Tensor, ...]
    split: dict[str, int]
    data_offset: int
    identity: tuple[int, ...]


def _identity(info: os.stat_result) -> tuple[int, ...]:
    # The fields that an open file and its path report alike on every OS; on
    # Windows, os.fstat and os.stat give st_ctime different meanings.
    return info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns


def _tensor_size(name: str, shape: tuple[int, ...], type_id: int) -> int:
    if type_id not in NATIVE_TYPES:
        raise ValueError(f"tensor {name} has unsupported native ggml type {type_id}")
    _, block, size = NATIVE_TYPES[type_id]
    if not shape or len(shape) > 4 or any(d <= 0 or d > MAX_I64 for d in shape):
        raise ValueError(f"tensor {name} has invalid dimensions {shape}")
    if shape[0] % block:
        raise ValueError(f"tensor {name}: quantized row size is not a multiple of block {block}")
    elements = 1
    for dimension in shape:
        if dimension > MAX_I64 // elements:
            raise ValueError(f"tensor {name}: element count overflows int64")
        elements *= dimension
    if elements // block > MAX_I64 // size:
        raise ValueError(f"tensor {name}: byte count overflows int64")
    return elements // block * size


def read_header(path: Path) -> Header:
    resolved = path.resolve(strict=True)
    if any(char in str(resolved) for char in "\t\r\n"):
        raise RuntimeError("expert source path cannot contain tabs or newlines")
    descriptor = os.open(resolved, OPEN_FLAGS)
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode) or not 0 <= before.st_size <= MAX_I64:
            raise ValueError("GGUF source must be a regular file with an int64 length")
        # Unbuffered reads keep logical I/O within header fields; no mmap or
        # buffered read-ahead reaches into the tensor data region.
        stream = os.fdopen(descriptor, "rb", buffering=0)
        descriptor = -1
        with stream:
            reader = _Reader(stream, before.st_size)
            if reader.read(4) != b"GGUF":
                raise ValueError("GGUF magic invalid")
            version_bytes = reader.read(4)
            version = struct.unpack("<I", version_bytes)[0]
            if version not in (2, 3):
                if struct.unpack(">I", version_bytes)[0] in (2, 3):
                    raise ValueError("big-endian GGUF is not supported by the alias builder")
                raise ValueError(f"unsupported GGUF version {version}")
            tensor_count, metadata_count = int(reader.take("Q")), int(reader.take("Q"))
            if metadata_count > (reader.length - reader.offset) // 12:
                raise ValueError("truncated GGUF metadata table")
            # GGUFReader reserves these synthetic fields before reading KV.
            keys = {"GGUF.version", "GGUF.tensor_count", "GGUF.kv_count"}
            split: dict[str, int] = {}
            alignment = 32
            for _ in range(metadata_count):
                key, kind = reader.string(), int(reader.take("I"))
                if key in keys:
                    raise ValueError(f"duplicate GGUF metadata key: {key}")
                keys.add(key)
                if key == "general.alignment" and kind != 4:
                    raise ValueError("Bad type for general.alignment field")
                expected_split_type = {"split.no": 2, "split.count": 2, "split.tensors.count": 5}
                if key in expected_split_type and kind != expected_split_type[key]:
                    raise ValueError(f"Bad type for {key} field")
                value = reader.value(kind)
                if key == "general.alignment":
                    alignment = int(value)
                    if alignment == 0 or alignment & (alignment - 1):
                        raise ValueError("Invalid alignment: must be a non-zero power of two")
                if key in expected_split_type:
                    split[key] = int(value)
            if tensor_count > (reader.length - reader.offset) // 24:
                raise ValueError("truncated GGUF tensor table")
            tensors = []
            names: set[str] = set()
            for _ in range(tensor_count):
                # Native gguf rejects every tensor name at GGML_MAX_NAME (128)
                # bytes, including dense tensors not selected for the alias.
                name, dimensions = reader.string(max_bytes=127), int(reader.take("I"))
                if name in names:
                    raise ValueError(f"Found duplicated tensor with name {name}")
                names.add(name)
                if not 1 <= dimensions <= 4:
                    raise ValueError(f"tensor {name} has invalid dimension count {dimensions}")
                shape = tuple(int(reader.take("Q")) for _ in range(dimensions))
                type_id, offset = int(reader.take("I")), int(reader.take("Q"))
                tensors.append(
                    Tensor(name, shape, type_id, offset, _tensor_size(name, shape, type_id))
                )
            data_offset = (reader.offset + alignment - 1) // alignment * alignment
            # Native GGUF permits metadata-only shards to end immediately
            # after their header. Seeking to their logical aligned data start
            # can pass EOF, but there is no tensor range or payload to read.
            if tensors and data_offset > reader.length:
                raise ValueError("GGUF data region starts beyond the source file")
            checked = []
            expected_offset = 0
            for tensor in tensors:
                if (
                    tensor.offset > reader.length - data_offset
                    or tensor.nbytes > reader.length - data_offset - tensor.offset
                ):
                    raise ValueError(f"tensor {tensor.name}: payload range exceeds source file")
                # gguf_init_from_file requires header order to describe one
                # contiguous, alignment-padded data section. Range checks alone
                # would admit overlapping tensors or arbitrary gaps.
                if tensor.offset != expected_offset:
                    raise ValueError(
                        f"tensor {tensor.name}: offset {tensor.offset}, "
                        f"expected native ordered offset {expected_offset}"
                    )
                padded_size = (tensor.nbytes + alignment - 1) // alignment * alignment
                if padded_size > MAX_I64 - expected_offset:
                    raise ValueError(f"tensor {tensor.name}: padded data size overflows int64")
                expected_offset += padded_size
                checked.append(
                    Tensor(
                        tensor.name,
                        tensor.shape,
                        tensor.type_id,
                        data_offset + tensor.offset,
                        tensor.nbytes,
                    )
                )
            if _identity(before) != _identity(os.fstat(stream.fileno())) or _identity(
                before
            ) != _identity(resolved.stat()):
                raise RuntimeError("GGUF source changed while reading its header")
            return Header(resolved, tuple(checked), split, data_offset, _identity(before))
    finally:
        if descriptor >= 0:
            os.close(descriptor)


def parse_model_paths(patterns: list[str]) -> list[Path]:
    result, seen = [], set()
    for pattern in patterns:
        matches = sorted(glob.glob(pattern))
        if not matches:
            raise RuntimeError(f"model glob matched no files: {pattern}")
        for match in matches:
            path = Path(match)
            resolved = path.resolve()
            if resolved not in seen:
                seen.add(resolved)
                result.append(path)
    return result


def _validate_shards(paths: list[Path]) -> list[int]:
    if not paths:
        raise RuntimeError("no GGUF model files were selected")
    resolved = [path.resolve(strict=True) for path in paths]
    if len(set(resolved)) != len(resolved):
        raise RuntimeError("duplicate GGUF source file")
    first = SHARD_RE.fullmatch(resolved[0].name)
    if first is None:
        if len(paths) != 1:
            raise RuntimeError("multiple GGUF files must form one complete numbered shard set")
        return [0]
    prefix, first_number, total_text = first.groups()
    total = int(total_text)
    if first_number != "00001" or total < 1:
        raise RuntimeError("model must name the first GGUF shard")
    if len(paths) != total:
        raise RuntimeError("GGUF source list is not the complete numbered shard set")
    expected = {
        resolved[0].parent / f"{prefix}-{i:05d}-of-{total_text}.gguf" for i in range(1, total + 1)
    }
    if set(resolved) != expected:
        raise RuntimeError("GGUF source list is not the complete numbered shard set")
    numbers = []
    for path in resolved:
        match = SHARD_RE.fullmatch(path.name)
        if match is None:
            raise RuntimeError("GGUF source list contains an unnumbered shard")
        numbers.append(int(match.group(2)) - 1)
    return numbers


def build_rows(
    model_paths: list[Path],
    include_kinds: set[str],
    n_experts: int | None,
    require_aligned: bool = False,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Index every selected expert tensor. Each has the same expert count: n_experts, or the
    first selected tensor's when it is None."""
    if n_experts is not None and n_experts < 1:
        raise RuntimeError("the expert count must be positive")
    if not include_kinds or include_kinds - {"up", "gate", "down"}:
        raise RuntimeError("include kinds must be a nonempty subset of up,gate,down")
    shard_numbers = _validate_shards(model_paths)
    headers = [read_header(path) for path in model_paths]
    total_tensors = sum(len(header.tensors) for header in headers)
    rows: list[dict[str, Any]] = []
    seen: set[str] = set()
    all_tensor_names: set[str] = set()
    by_kind: dict[str, dict[str, int]] = defaultdict(lambda: dict(tensors=0, entries=0, bytes=0))
    by_type: dict[str, dict[str, int]] = defaultdict(lambda: dict(tensors=0, entries=0, bytes=0))
    unaligned_rows = total_bytes = 0
    for shard_index, (header, shard_number) in enumerate(zip(headers, shard_numbers)):
        # Match the native loader's required split metadata. Later shards may
        # omit the count fields, but their own split.no is always required.
        if len(headers) > 1:
            required = {"split.no"}
            if shard_index == 0:
                required |= {"split.count", "split.tensors.count"}
            if not required.issubset(header.split):
                raise RuntimeError(f"{header.path}: missing GGUF split metadata")
        expected_split = {
            "split.no": shard_number,
            "split.count": len(headers),
            "split.tensors.count": total_tensors,
        }
        for key, value in header.split.items():
            if value != expected_split[key]:
                raise RuntimeError(f"{header.path}: inconsistent {key} metadata")
        for tensor in header.tensors:
            match = EXPERT_RE.fullmatch(tensor.name)
            if tensor.name in all_tensor_names:
                if match is not None and match.group("kind") in include_kinds:
                    raise RuntimeError(f"duplicate expert tensor across shards: {tensor.name}")
                raise RuntimeError(f"duplicate tensor across shards: {tensor.name}")
            all_tensor_names.add(tensor.name)
            if match is None or match.group("kind") not in include_kinds:
                continue
            kind = match.group("kind")
            seen.add(tensor.name)
            if len(tensor.name.encode("utf-8")) >= 128:
                raise RuntimeError(f"tensor name too long for runtime key: {tensor.name}")
            count = tensor.shape[2] if len(tensor.shape) == 3 else 0
            if n_experts is None:
                n_experts = count
            if not count or count != n_experts:
                raise RuntimeError(
                    f"{tensor.name}: expected 3D tensor with {n_experts or 'routed'} experts, "
                    f"got {tensor.shape}"
                )
            if tensor.nbytes <= 0 or tensor.nbytes % n_experts:
                raise RuntimeError(
                    f"{tensor.name}: {tensor.nbytes} bytes not divisible by n_experts={n_experts}"
                )
            expert_bytes = tensor.nbytes // n_experts
            type_name = NATIVE_TYPES[tensor.type_id][0]
            for expert in range(n_experts):
                offset = tensor.offset + expert * expert_bytes
                aligned = offset % ALIGNMENT == 0 and expert_bytes % ALIGNMENT == 0
                if not aligned:
                    unaligned_rows += 1
                    if require_aligned:
                        raise RuntimeError(
                            f"{tensor.name}:{expert}: unaligned offset={offset} "
                            f"nbytes={expert_bytes}"
                        )
                rows.append(
                    dict(
                        tensor=tensor.name,
                        expert_idx=expert,
                        nbytes=expert_bytes,
                        source_path=str(header.path),
                        offset=offset,
                        kind=kind,
                        layer=int(match.group("layer")),
                        type=type_name,
                        aligned=int(aligned),
                        shard_index=shard_index,
                    )
                )
            for bucket in (by_kind[kind], by_type[type_name]):
                bucket["tensors"] += 1
                bucket["entries"] += n_experts
                bucket["bytes"] += tensor.nbytes
            total_bytes += tensor.nbytes
    if not rows:
        raise RuntimeError("no expert tensors matched the requested kinds")
    if any(header.identity != _identity(header.path.stat()) for header in headers):
        raise RuntimeError("GGUF source changed while building its alias index")
    rows.sort(key=lambda row: (str(row["tensor"]), int(row["expert_idx"]), int(row["nbytes"])))
    summary = dict(
        model_paths=[str(path) for path in model_paths],
        model_shards=len(model_paths),
        expert_tensors=len(seen),
        entries=len(rows),
        n_experts=n_experts,
        include_kinds=sorted(include_kinds),
        total_payload_bytes=total_bytes,
        total_payload_gib=total_bytes / 1024**3,
        unaligned_rows=unaligned_rows,
        by_kind=dict(sorted(by_kind.items())),
        by_type=dict(sorted(by_type.items())),
    )
    return rows, summary


def _repo_head(repo: Path | None) -> str:
    override = os.environ.get("SSD_LLAMA_REPO_HEAD")
    if override:
        return override
    if repo is None:
        return "unknown"
    try:
        result = run_captured(["git", "-C", str(repo), "rev-parse", "HEAD"], timeout=10)
    except (OSError, subprocess.SubprocessError):
        return "unknown"
    return result.stdout.strip() if result.returncode == 0 else "unknown"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-glob", action="append", required=True)
    parser.add_argument("--n-experts", type=int, help="expected experts per tensor (default: read)")
    parser.add_argument("--include-kind", default="up,gate,down")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--summary-json", type=Path, required=True)
    parser.add_argument("--repo", type=Path, help="optional explicit source provenance only")
    parser.add_argument("--require-aligned", action="store_true")
    args = parser.parse_args(argv)
    kinds = {kind.strip() for kind in args.include_kind.split(",") if kind.strip()}
    paths = parse_model_paths(args.model_glob)
    rows, summary = build_rows(paths, kinds, args.n_experts, args.require_aligned)
    summary.update(repo_head=_repo_head(args.repo), output_tsv=str(args.out))
    # The caller supplies a private staging directory and publishes only after
    # this process has validated every shard/header/tensor and returned success.
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDS, delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)
    args.summary_json.parent.mkdir(parents=True, exist_ok=True)
    args.summary_json.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(f"entries={summary['entries']}")
    print(f"expert_tensors={summary['expert_tensors']}")
    print(f"payload_gib={summary['total_payload_gib']:.3f}")
    print(f"unaligned_rows={summary['unaligned_rows']}")
    print(f"out={args.out}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as error:
        print(f"error: {error}", file=sys.stderr)
        raise SystemExit(1)

"""Synthetic screenshots for vision deployments, as PNG data URIs.

A vision request costs what a text one does not: the encoder runs once per
image the server has not cached, and the image's tokens join the prompt. The
probe needs images whose SIZE is the workload's and whose BYTES are new to
the server, since vLLM keys its prefix and encoder caches on a hash of the
image: an image the server saw last run is a hit whatever the text says.

Each image is a black frame whose bottom row is random pixels drawn from a
64-bit id, so an image is unique per id and reproducible from it. The black
rows compress to a few kilobytes on the wire while the server still decodes
the full resolution, and they are compressed once per size: the images are
built on the event loop that times every stream, and compressing a full
1280x800 frame per image cost ~8 ms of client stall each. stdlib only.
"""
from __future__ import annotations

import base64
import functools
import random
import struct
import zlib


def _chunk(tag: bytes, data: bytes) -> bytes:
    return (struct.pack(">I", len(data)) + tag + data
            + struct.pack(">I", zlib.crc32(tag + data) & 0xFFFFFFFF))


@functools.lru_cache(maxsize=8)
def _black_rows(width: int, height: int):
    """A compressor that has consumed every scanline but the last, all black
    (filter byte 0 and zero pixels). Copied, never fed, by `png_bytes`."""
    z = zlib.compressobj(6)
    head = z.compress(bytes((1 + 3 * width) * (height - 1)))
    return head, z


def png_bytes(image_id: int, width: int, height: int) -> bytes:
    """An 8-bit RGB PNG, `width` x `height`, a pure function of `image_id`."""
    head, z = _black_rows(width, height)
    z = z.copy()
    row = random.Random(image_id).randbytes(3 * width)
    idat = head + z.compress(b"\x00" + row) + z.flush()
    ihdr = struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)
    return (b"\x89PNG\r\n\x1a\n" + _chunk(b"IHDR", ihdr)
            + _chunk(b"IDAT", idat) + _chunk(b"IEND", b""))


def data_uri(image_id: int, width: int, height: int) -> str:
    return ("data:image/png;base64,"
            + base64.b64encode(png_bytes(image_id, width, height)).decode())

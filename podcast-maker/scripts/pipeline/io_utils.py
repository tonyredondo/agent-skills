#!/usr/bin/env python3
from __future__ import annotations

"""I/O helpers shared by pipeline entrypoints."""

from typing import Callable, Optional, Tuple


def read_text_file_with_fallback(
    path: str,
    *,
    on_fallback: Optional[Callable[[str], None]] = None,
) -> Tuple[str, str]:
    """Read text file trying a safe sequence of fallback encodings.

    Returns `(content, encoding_used)`.
    """
    encodings = ["utf-8", "utf-8-sig", "cp1252", "latin-1"]
    last_exc: Exception | None = None
    for enc in encodings:
        try:
            with open(path, "r", encoding=enc) as f:
                data = f.read()
            if enc != "utf-8" and on_fallback is not None:
                on_fallback(enc)
            return data, enc
        except Exception as exc:  # noqa: BLE001
            last_exc = exc
            continue
    raise RuntimeError(f"Failed to decode input file with supported encodings: {last_exc}")


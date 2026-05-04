from __future__ import annotations

from typing import Final

class VerifiedTrace:
    format_version: int
    recorder_version: str
    canonicalisation_version: str
    price_list_version: str
    public_key_hex: str
    hmac_key_id: str
    frame_count: int
    def __repr__(self) -> str: ...

class VerifyError(Exception):
    kind: str
    frame_index: int

def verify_bytes(buf: bytes, hmac_key: bytes) -> VerifiedTrace: ...
def verify_path(path: str, hmac_key: bytes) -> VerifiedTrace: ...

__version__: Final[str]

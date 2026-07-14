from __future__ import annotations

import hashlib


def feature_list_hash(feature_cols: list[str]) -> str:
    payload = "\n".join(feature_cols).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()[:16]

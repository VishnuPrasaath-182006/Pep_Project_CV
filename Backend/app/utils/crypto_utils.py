import hashlib
import os
from pathlib import Path
from typing import Dict, Tuple, Optional

# In-memory checksum cache: (path_str, size, mtime) -> sha256_hex
_CHECKSUM_CACHE: Dict[Tuple[str, int, float], str] = {}

def compute_file_sha256(file_path: Path, chunk_size: int = 65536) -> Optional[str]:
    """
    Computes streaming SHA-256 digest of a file in 64 KB chunks to ensure low memory footprint.
    Caches results keyed by file modification time and size.
    """
    path_obj = Path(file_path)
    if not path_obj.exists() or not path_obj.is_file():
        return None
        
    try:
        stat = path_obj.stat()
        cache_key = (str(path_obj.resolve()), stat.st_size, stat.st_mtime)
        if cache_key in _CHECKSUM_CACHE:
            return _CHECKSUM_CACHE[cache_key]
            
        hasher = hashlib.sha256()
        with open(path_obj, "rb") as f:
            while chunk := f.read(chunk_size):
                hasher.update(chunk)
                
        digest = hasher.hexdigest()
        _CHECKSUM_CACHE[cache_key] = digest
        return digest
    except Exception:
        return None

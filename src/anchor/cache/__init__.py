"""Caching module for anchor pipeline steps."""

from .backend import InMemoryCacheBackend
from .sqlite_backend import SqliteCacheBackend

__all__ = [
    "InMemoryCacheBackend",
    "SqliteCacheBackend",
]

# Redis backend is optional (requires redis-py)
try:
    from .redis_backend import RedisCacheBackend
    __all__.append("RedisCacheBackend")
except ImportError:
    import importlib.util as _iu
    if _iu.find_spec("redis") is not None:
        raise  # redis installed but import still failed — surface the real error
    del _iu

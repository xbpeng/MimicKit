import os as _os
import sys as _sys

# MimicKit's internal modules use flat imports (e.g. ``import learning.ppo_agent``).
# When installed as a package, these submodules live under ``mimickit/`` which is
# not on sys.path by default.  Adding it here keeps backward compatibility with
# the original ``python mimickit/run.py`` invocation while also supporting
# ``import mimickit`` from an installed environment.
_pkg_dir = _os.path.dirname(__file__)
if _pkg_dir not in _sys.path:
    _sys.path.insert(0, _pkg_dir)

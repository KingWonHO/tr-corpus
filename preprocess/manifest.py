# sha256 inventories of input files, keyed by their place under configs/, data/ or results/
import os
from pathlib import Path

from core.fitting import sha


def roots(paths):
    return {'configs': paths.configs, 'data': paths.data, 'results': paths.results}


# 'configs/...', 'data/...' or 'results/...' with forward slashes
def key(paths, p):
    p = Path(os.path.abspath(p))
    for prefix, root in roots(paths).items():
        try:
            return prefix + '/' + p.relative_to(os.path.abspath(root)).as_posix()
        except ValueError:
            continue
    raise ValueError('%s is outside configs, data and results' % p)


def resolve(paths, name):
    prefix, rest = name.split('/', 1)
    return Path(roots(paths)[prefix]) / rest


def hashes(paths, files):
    return {key(paths, p): sha(p) for p in files if Path(p).is_file()}

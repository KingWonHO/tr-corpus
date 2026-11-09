# base class: one experiment reads earlier results and writes results/<name>
import os

os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')

import torch  # noqa: E402


class Experiment:
    name = ''
    help = ''

    def __init__(self, paths, args):
        self.paths = paths
        self.args = args
        self.out = paths.out(self.name)
        torch.set_num_threads(4)

    # experiment-specific command-line options
    @staticmethod
    def add_arguments(ap):
        pass

    def run(self):
        raise NotImplementedError


def split_arg(value, cast=str):
    return [cast(v) for v in value.split(',')] if value else []

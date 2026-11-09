# data and results locations shared by every step
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


class Paths:
    def __init__(self, data='data/tr-corpus', raw='data/raw', results='results'):
        self.data = Path(data)
        self.raw = Path(raw)
        self.results = Path(results)
        self.configs = ROOT / 'configs'
        self.registry = self.data / 'registry'
        self.windows = self.data / 'windows' / 'W60_native'
        self.measured = self.data / 'measured'
        self.arc_raw = self.raw / 'ds12_arc'

    # results/<name>, created on first use
    def out(self, name):
        p = self.results / name
        p.mkdir(parents=True, exist_ok=True)
        return p

    # results/<name> of an earlier step, which must exist
    def need(self, name):
        p = self.results / name
        if not p.is_dir():
            raise FileNotFoundError('run the step that writes %s first' % p)
        return p

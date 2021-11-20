from pathlib import Path
import sys

from importlib import reload


NOTEBOOKS_DIR = Path(__file__).parent
ROOT_DIR = (NOTEBOOKS_DIR / '..').resolve()

sys.path.append(str(ROOT_DIR))


def get_reloader(*modules):
    def reloader():
        for m in modules:
            reload(m)
    
    return reloader

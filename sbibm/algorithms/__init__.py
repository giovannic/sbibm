import os

os.environ.setdefault("XDG_CACHE_HOME", "/tmp")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

from sbibm.algorithms.deepset import run as deepset
from sbibm.algorithms.simformer import run as simformer
from sbibm.algorithms.sbi.snle import run as snle
from sbibm.algorithms.sbi.snpe import run as snpe
from sbibm.algorithms.sbi.snre import run as snre

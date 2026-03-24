import os

os.environ.setdefault("XDG_CACHE_HOME", "/tmp")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

from sbibm.algorithms.deepset import run as deepset
from sbibm.algorithms.sbi.mcabc import run as mcabc
from sbibm.algorithms.sbi.smcabc import run as smcabc
from sbibm.algorithms.simformer import run as simformer
from sbibm.algorithms.sbi.snle import run as snle
from sbibm.algorithms.sbi.snpe import run as snpe
from sbibm.algorithms.sbi.snre import run as snre

rej_abc = mcabc
smc_abc = smcabc

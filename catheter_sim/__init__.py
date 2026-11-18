"""2D steerable-catheter navigation simulator."""
import os

# The solver works on small dense matrices; multithreaded BLAS only adds
# overhead (and can be >10x slower). Must be set before numpy is imported.
for _v in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

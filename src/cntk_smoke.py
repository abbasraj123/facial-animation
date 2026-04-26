import os
import pathlib as plb
import sys

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("OMP_NUM_THREADS", "1")
cntk_libs = plb.Path(sys.prefix) / "Lib" / "site-packages" / "cntk" / "libs"
if cntk_libs.exists():
    os.environ["PATH"] = str(cntk_libs) + os.pathsep + os.environ.get("PATH", "")

import cntk as C
import numpy as np

print("start")
x = C.input_variable((2,))
y = x + 1
print(y.eval({x: np.array([[1, 2]], dtype=np.float32)}))

import os, sys
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE)); sys.path.insert(0, HERE)
import torch, torch_npu  # noqa
from gqa_flat_variant import build_flat_kernel
try:
    fn = build_flat_kernel(2,128,128,1.0/128**0.5,True,32,8,128,'float16')
    print('COMPILE OK')
except Exception as e:
    s = str(e)
    print('EXC LEN', len(s))
    # print the bisheng error portion
    import re
    print(s[:8000])

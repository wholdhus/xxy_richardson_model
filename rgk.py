import time
import sys
from tests import test_rgk


if __name__ == '__main__':
    start = time.time()
    L = int(sys.argv[1])
    N = int(sys.argv[2])
    g_step = float(sys.argv[3])/L
    test_rgk(L, N, g_step)
    finish = time.time()
    print('Seconds elapsed: {}'.format(finish-start))

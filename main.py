import os
import numpy as np
from pygears import gear  # , sim, reg
from pygears.lib import dreg, qround, saturate, drv, collect
from pygears.typing import Fixp, Tuple, Array
from channel import RayleighChannel, AWGNChannel


@gear
def fir_direct(din, b):
    temp = din
    add_s = temp * b[0]
    for coef in b[1:]:
        temp = temp | dreg(init=0)
        add_s = add_s + (temp * coef)
    return add_s | qround(fract=din.dtype.fract) | saturate(t=din.dtype)


def test():
    # generate test sequence
    raylChan = RayleighChannel(mean_delay=50, max_delay=100, rician_factor=1.0, var_rate=0.01)
    awgnChan = AWGNChannel(pwr=0.001)
    xs_rx = [awgnChan(raylChan(np.random.randint(2) * 2 - 1)) for _ in range(100)]
    b = Array[Fixp[3, 8], 4](val=tuple([0.0] * 4))

    # setup simulation
    res = []
    # reg["debug/trace"] = ['*']
    drv(t=Fixp[3, 8], seq=xs_rx) \
    | fir_direct(b) \
    | collect(result=res)
    # sim(resdir='/work/', timeout=128, seed=1234)
    print(res)


if __name__ == '__main__':
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    # print(os.getcwd())
    test()

import os
import numpy as np
from pygears import gear, sim, reg
from pygears.lib import drv, collect
from pygears.typing import Fixp
from dataflow_template import fir_adaptive, fir_adaptive_top, psk_quantizer
from channel import RayleighChannel, AWGNChannel


def fir_direct_testbench():
    # generate test sequence
    raylChan = RayleighChannel(mean_delay=50, max_delay=100, rician_factor=1.0, var_rate=0.01)
    awgnChan = AWGNChannel(pwr=0.001)
    xs_rx = [awgnChan(raylChan(np.random.randint(2) * 2 - 1)) for _ in range(100)]
    
    # set architecture parameters
    wl_fixp  = 8
    wl_int   = 3
    wl_fract = wl_fixp - wl_int
    num_tap  = 4
    init_val = tuple([1.0] + [0.0] * (num_tap-1))
    
    # filter weights
    b = Array[Fixp[wl_int, wl_fixp], num_tap](val=init_val)
    
    # setup simulation
    res = []
    reg["debug/trace"] = ['*']
    drv(t=Fixp[wl_int, wl_fixp], seq=xs_rx) \
    | fir_direct(b=b, fract=wl_fract) \
    | collect(result=res)
    sim(resdir='../sim/', timeout=128, seed=1234)
    
    for xs, r in zip(xs_rx, res):
    	print(xs, float(r))
    return
    
   
def fir_adaptive_testbench():
    # generate test sequence
    raylChan = RayleighChannel(mean_delay=1, max_delay=4, rician_factor=1.0, var_rate=0.00)
    awgnChan = AWGNChannel(pwr=0.000)
    xs_tx = []
    xs_rx = []
    
    for i in range(1000):
        x_tx = np.random.randint(2) * 2.0 - 1.0
        x_rx = awgnChan(raylChan(x_tx))
        xs_tx.append(x_tx)
        xs_rx.append(x_rx)
    
    # set architecture parameters
    wl_fixp  = 16
    wl_int   = 3
    wl_fract = wl_fixp - wl_int
    num_tap  = 4
    init_coeffs = tuple([1.0] + [0.0] * (num_tap-1))
    
    # setup simulation
    res = []
    reg["debug/trace"] = ['*']
    drv_din = drv(t=Fixp[wl_int, wl_fixp], seq=xs_rx)
    drv_dtarget = drv(t=Fixp[wl_int, wl_fixp], seq=xs_tx)
    
    fir_adaptive_top(din=drv_din, dtarget=drv_dtarget, init_coeffs=init_coeffs, 
                     lr=0.01, quantizer=psk_quantizer) \
    	| collect(result=res)
    sim(resdir='../sim/')
    
    for x_tx, x_rx, r in zip(xs_tx, xs_rx, res):
    	print(f"{x_tx:.2f}, {x_rx:.2f}, {float(r):.2f}")
    return


if __name__ == '__main__':
    # os.chdir(os.path.dirname(os.path.abspath(__file__)))
    # print(os.getcwd())
    # fir_direct_testbench()
    fir_adaptive_testbench()
    
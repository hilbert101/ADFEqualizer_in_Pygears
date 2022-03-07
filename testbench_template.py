import os
import numpy as np
from matplotlib import pyplot as plt
from pygears import gear, Intf, sim, reg
from pygears.lib import drv, collect, ccat, const
from pygears.typing import Fixp, Uint
from dataflow_template import psk_quantizer
from dataflow_template import fir_direct, fir_adaptive, fir_adaptive_top, dfe_adaptive_top
from dataflow_template import dfe_adaptive_wrap
from pygears.hdl import hdlgen
from adfe_util import pam4_quantizer
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
    num_tap  = 3
    dtype    = Fixp[wl_int, wl_fixp]
    
    # filter weights
    #b = Array[Fixp[wl_int, wl_fixp], num_tap](val=init_val)
    b = [1.0, 0.5, 0.25]
    
    # setup simulation
    res = []
    reg["debug/trace"] = ['*']
    x_drv = drv(t=dtype, seq=xs_rx)
    b_drv = ccat(*[const(val=coef, tout=dtype) for coef in b])
    
    fir_direct(x_drv, b_drv) \
    | collect(result=res)
    sim(resdir='../sim/', timeout=128, seed=1234)
    
    for xs, r in zip(xs_rx, res):
    	print(xs, float(r))
    return
    
   
def fir_adaptive_testbench():
    # generate test sequence
    raylChan = RayleighChannel(mean_delay=2, max_delay=8, rician_factor=10.0, var_rate=0.00)
    awgnChan = AWGNChannel(pwr=0.000)
    xs_tx = []
    xs_rx = []
    
    for i in range(1000):
        x_tx = np.random.randint(4) * 2.0 - 3.0
        x_rx = awgnChan(raylChan(x_tx))
        xs_tx.append(x_tx)
        xs_rx.append(x_rx)
    
    # set architecture parameters
    wl_fixp  = 16
    wl_int   = 5
    wl_fract = wl_fixp - wl_int
    dtype    = Fixp[wl_int, wl_fixp]
    num_tap  = 16
    init_coeffs = tuple([1.0] + [0.0] * (num_tap-1))
    
    # setup simulation
    res = []
    reg["debug/trace"] = [] # ['*'] # use ['*'] for debug (but sim takes longer)
    drv_din = drv(t=Fixp[wl_int, wl_fixp], seq=xs_rx)
    drv_dtarget = drv(t=Fixp[wl_int, wl_fixp], seq=xs_tx)
    
    fir_adaptive_top(din=drv_din, dtarget=drv_dtarget, init_coeffs=init_coeffs, 
                     lr=0.003, quantizer=psk_quantizer) \
        | collect(result=res)
    sim(resdir='../sim/')
    
    #for x_tx, x_rx, (d, e) in zip(xs_tx, xs_rx, res):
    #    print(f"{x_tx:.2f}, {x_rx:.2f}, {float(e):.2f}")
    es = np.abs(np.array([float(e) for d, e in res]))
    #print(np.array(xs_tx))
    #print(np.array([float(e) for d, e in res]))
    #print(es)
    plt.plot(np.arange(len(es)), es)
    plt.show()
    return
    
    
def dfe_adaptive_testbench():
    # generate test sequence
    raylChan = RayleighChannel(mean_delay=2, max_delay=8, rician_factor=10.0, var_rate=0.00)
    awgnChan = AWGNChannel(pwr=0.000)
    xs_tx = []
    xs_rx = []
    
    len_train = 500  # number of data for training
    len_track = 500  # number of data for blind tracking
    for i in range(len_train + len_track):
        x_tx = np.random.randint(4) * 2.0 - 3.0
        x_rx = awgnChan(raylChan(x_tx))
        xs_tx.append(x_tx)
        xs_rx.append(x_rx)
    
    # set architecture parameters
    wl_fixp  = 16
    wl_int   = 5
    wl_fract = wl_fixp - wl_int
    fftap    = 12
    fbtap    = 8
    init_ff_coeffs = tuple([1.0] + [0.0] * (fftap-1))
    init_fb_coeffs = tuple([0.0] * (fbtap))
    
    # setup simulation
    res = []
    reg["debug/trace"] = []
    drv_din = drv(t=Fixp[wl_int, wl_fixp], seq=xs_rx)
    drv_dtarget = drv(t=Fixp[wl_int, wl_fixp], seq=xs_tx[:len_train] + [0]*len_track)
    
    dfe_adaptive_wrap(din=drv_din, dtarget=drv_dtarget, \
                      init_ff_coeffs=init_ff_coeffs, init_fb_coeffs=init_fb_coeffs, \
                      lr=0.003, quantizer=pam4_quantizer) \
        | collect(result=res)
    sim(resdir='../sim/')
    
    #for x_tx, x_rx, r in zip(xs_tx, xs_rx, res):
    #    print(f"{x_tx:.2f}, {x_rx:.2f}, {float(r):.2f}")
    es = np.abs(np.array([float(e) for d, e in res]))
    plt.plot(np.arange(len(es)), es)
    plt.show()
    return


def adfe_hdlgen():
    # set architecture parameters
    wl_fixp  = 16
    wl_int   = 5
    wl_fract = wl_fixp - wl_int
    ff_tap   = 12
    fb_tap   = 8
    
    init_ff_coeffs = tuple([1.0] + [0.0] * (ff_tap-1))
    init_fb_coeffs = tuple([0.0] * fb_tap)
    
    dtype    = Fixp[wl_int, wl_fixp]
    
    din = Intf(dtype)
    dtarget = Intf(dtype)
    dfe_adaptive_wrap(din=din, dtarget=dtarget, \
                      init_ff_coeffs=init_ff_coeffs, init_fb_coeffs=init_fb_coeffs, \
                      lr=0.003, quantizer=pam4_quantizer)
    hdlgen(top='/dfe_adaptive_wrap', lang='sv', outdir='../HDL/adfe_org_top', copy_files=True)
    
    """
        CK = 5.0 ns, power = 1.06 mW, area = 8562.162731 um^2, EDP = 
        CK = 3.0 ns, power = 1.40 mW, area = 7272.964183 um^2, EDP = 
        CK = 2.5 ns, power = 1.64 mW, area = 7534.412747 um^2, EDP = 
        CK = 2.4 ns, power = 1.68 mW, area = 7640.409420 um^2, EDP =
        CK = 2.3 ns, x
    """ 


if __name__ == '__main__':
    # os.chdir(os.path.dirname(os.path.abspath(__file__)))
    # print(os.getcwd())
    fir_direct_testbench()
    # fir_adaptive_testbench()
    # dfe_adaptive_testbench()
    # adfe_hdlgen()
    

import os
import numpy as np
from matplotlib import pyplot as plt
from pygears import gear, Intf, sim, reg
from pygears.lib import dreg, decouple, const, ccat, qround, saturate
from pygears.lib import flatten, priority_mux, replicate, once, union_collapse
from pygears.lib import drv, collect
from pygears.typing import Int, Uint, Fixp, Tuple, Array, ceil_pow2
from pygears.hdl import hdlgen
from fir_opt_retime import fir_opt_retime
from adfe_fb import adfe_fb_stag, adfe_fb_stag_v1
from adfe_lms import adfe_ff_adapt_coeffs, adfe_fb_adapt_luts
from adfe_util import qam16_quantizer, decouple_reg, mux_comb
from testbench_template import fir_adaptive_testbench
from channel import RayleighChannel, AWGNChannel


@gear
def fir_opt_retime_adapt(din, dtarget, *, init_coeffs=(1.0, ), lr=0.01):
    din = din | dreg(init=0)
    dtarget = dtarget | dreg(init=0) | dreg(init=0) | dreg(init=0)
    dpred = Intf(din.dtype)
    
    dquant = qam16_quantizer(dpred)[0]
    ctrl = (dtarget == const(val=0.0, tout=din.dtype)) # control for training/tracking
    dsel = mux_comb(ctrl, ccat(dtarget, dquant))  # self-made gears
    err = (dsel - dpred) | dreg(init=0)
    
    lr_err = const(val=lr, tout=din.dtype) * err \
        | qround(fract=din.dtype.fract) \
        | saturate(t=din.dtype) \
        | dreg(init=0)
    coeffs = adfe_ff_adapt_coeffs(din, lr_err, init_coeffs=init_coeffs, extra_latency=4)
    
    dpred |= fir_opt_retime(din, coeffs) | decouple_reg(init=0, num=1) 
    return ccat(dquant | dreg(init=0), err | dreg(init=0))  
    
"""    
@gear
def adfe_opt_v1(din, dtarget, *, init_ff_coeffs=(1.0, ), init_fb_coeffs=(1.0, ), lr=0.01):
    din = din | dreg(init=0)
    dtarget = dtarget | dreg(init=0) | dreg(init=0) | dreg(init=0)
    
    dpred  = Intf(din.dtype)
    dquant = qam16_quantizer(dpred)[0]
    ctrl   = (dtarget == const(val=0.0, tout=din.dtype)) # control for training/tracking
    dsel   = mux_comb(ctrl, ccat(dtarget, dquant))  # self-made gears
    
    err    = (dsel - dpred) | dreg(init=0)
    
    lr_err = const(val=lr, tout=din.dtype) * err \
        | qround(fract=din.dtype.fract) \
        | saturate(t=din.dtype) \
        | dreg(init=0)
        
    ff_coeffs = adfe_ff_adapt_coeffs(din, lr_err, 
                                     init_coeffs=init_ff_coeffs, extra_latency=3)
    #fb_luts   = adfe_fb_adapt_luts(din, lr_err, level=2,
    #                               init_coeffs=init_fb_coeffs, extra_latency=3)
    fb_coeffs = adfe_ff_adapt_coeffs(din, lr_err, 
                                     init_coeffs=init_fb_coeffs, extra_latency=3)
                                     
    print(fb_coeffs.dtype)
    ff = fir_opt_retime(din, ff_coeffs)
    fb = adfe_fb_stag_v1(dquant, fb_coeffs)
    dpred |= (ff + fb) | saturate(t=din.dtype) | decouple_reg(init=0, num=1) 
    
    return ccat(dquant, err)
"""
    
    
@gear
def adfe_opt_top(din, dtarget, *, init_ff_coeffs=(1.0, ), init_fb_coeffs=(1.0, ), lr=0.01):
    din = din | dreg(init=0)
    dtarget = dtarget | dreg(init=0) | dreg(init=0) | dreg(init=0)
    
    dpred, dquant = Intf(din.dtype), Intf(din.dtype)
    ctrl = (dtarget == const(val=0.0, tout=din.dtype)) # control for training/tracking
    dsel = mux_comb(ctrl, ccat(dtarget, dquant))  # self-made gears
    
    err  = (dsel - dpred) | dreg(init=0)
    
    lr_err = const(val=lr, tout=din.dtype) * err \
        | qround(fract=din.dtype.fract) \
        | saturate(t=din.dtype) \
        | dreg(init=0)
        
    ff_coeffs = adfe_ff_adapt_coeffs(din, lr_err, 
                                     init_coeffs=init_ff_coeffs, extra_latency=4)
    fb_luts   = adfe_fb_adapt_luts(din, lr_err, level=2,
                                   init_coeffs=init_fb_coeffs, extra_latency=3)
    ff = fir_opt_retime(din, ff_coeffs)
    dqaunt_back, dpred_back = adfe_fb_stag(ff, fb_luts)
    dquant |= dqaunt_back
    dpred  |= dpred_back
    
    return ccat(dquant | dreg(init=0) , err | dreg(init=0))


def fir_opt_retime_test():
    # generate test sequence
    test_len = 5
    xs  = [(np.random.randint(2) * 2 - 1) for _ in range(test_len)]
    b   = [1.0, 0.5, 0.25]
    
    # set architecture parameters
    wl_fixp  = 16
    wl_int   = 5
    num_tap  = 3
    dtype    = Fixp[wl_int, wl_fixp]
    
    # setup simulation
    res = []
    reg["debug/trace"] = [] # ['*']
    x_drv = drv(t=dtype, seq=xs)
    b_drv = ccat(*[const(val=coef, tout=dtype) for coef in b])
    
    fir_opt_retime(x_drv, b_drv) \
    | collect(result=res)
    sim(resdir='../sim/')
    
    for x, r in zip(xs, res):
    	print(x, float(r))
    #hdlgen(top='/fir_transpose', lang='v', outdir='../HDL', copy_files=True)
    return

    
def fir_opt_retime_adapt_test():
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
    num_tap  = 16
    init_coeffs = tuple([1.0] + [0.0] * (num_tap-1))
    
    # setup simulation
    res = []
    reg["debug/trace"] = [] # ['*'] # use ['*'] for debug (but sim takes longer)
    drv_din = drv(t=Fixp[wl_int, wl_fixp], seq=xs_rx)
    drv_dtarget = drv(t=Fixp[wl_int, wl_fixp], seq=xs_tx)
    
    fir_opt_retime_adapt(din=drv_din, dtarget=drv_dtarget, 
                         init_coeffs=init_coeffs, lr=0.001) \
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


def adfe_test():
    # generate test sequence
    raylChan = RayleighChannel(mean_delay=2, max_delay=8, rician_factor=10.0, var_rate=0.00)
    awgnChan = AWGNChannel(pwr=0.01)
    xs_tx = []
    xs_rx = []
    
    for i in range(2000):
        x_tx = np.random.randint(4) * 2.0 - 3.0
        x_rx = awgnChan(raylChan(x_tx))
        xs_tx.append(x_tx)
        xs_rx.append(x_rx)
    
    # set architecture parameters
    wl_fixp  = 16
    wl_int   = 5
    wl_fract = wl_fixp - wl_int
    ff_tap   = 12
    fb_tap   = 8
    init_ff_coeffs = tuple([1.0] + [0.0] * (ff_tap-1))
    init_fb_coeffs = tuple([0.0] * fb_tap)
    
    # setup simulation
    res = []
    reg["debug/trace"] = [] # ['*'] # use ['*'] for debug (but sim takes longer)
    drv_din = drv(t=Fixp[wl_int, wl_fixp], seq=xs_rx)
    drv_dtarget = drv(t=Fixp[wl_int, wl_fixp], seq=xs_tx)
    
    adfe_opt_top(din=drv_din, dtarget=drv_dtarget, 
                 init_ff_coeffs=init_ff_coeffs, init_fb_coeffs=init_fb_coeffs, lr=0.003) \
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
    adfe_opt_top(din=din, dtarget=dtarget, 
                 init_ff_coeffs=init_ff_coeffs, init_fb_coeffs=init_fb_coeffs, lr=0.003)
    hdlgen(top='/adfe_opt_top', lang='sv', outdir='../HDL/adfe_opt_top', copy_files=True)
    
    """
        CK = 2.0 ns, power = 2.95 mW, area = 11026.996317 um^2, EDP = 5.90
        CK = 1.5 ns, power = 3.82 mW, area = 11392.744275 um^2, EDP = 5.73
        
    """
    
if __name__ == '__main__':
    #fir_opt_retime_test()
    #fir_opt_retime_adapt_test()
    adfe_test()
    #adfe_hdlgen()


import os
import numpy as np
from pygears import gear, Intf, sim, reg
from pygears.lib import decouple, const, ccat, qround, saturate, dreg 
from pygears.lib import flatten, priority_mux, replicate, once, union_collapse
from pygears.lib import drv, collect
from pygears.typing import Int, Uint, Fixp, Tuple, Array, ceil_pow2, trunc, code
from pygears.hdl import hdlgen
#from adfe_util import decouple_reg as dreg
from adfe_util import decouple_reg, mux_comb


@gear
def adfe_ff_adapt_coeffs(din, lr_err, *, init_coeffs=(1.0, ), extra_latency=0):
    temp = din 
    for i in range(extra_latency):
        temp = temp | dreg(init=0)
    
    coeffs = []
    for i, init_coeff in enumerate(init_coeffs):
        coeff = Intf(din.dtype)
        update = (lr_err * temp) | dreg(init=0)
        coeff_next = (coeff + update) \
            | qround(fract=din.dtype.fract) | saturate(t=din.dtype)
        coeff |= coeff_next | decouple_reg(init=init_coeff, num=1)
        coeffs.append(coeff)
        
        if i < len(init_coeffs)-1: # not the last
            temp = temp | dreg(init=0)
    
    return ccat(*coeffs)


@gear
def adfe_fb_adapt_luts(din, lr_err, *, level=2, init_coeffs=(1.0, ), extra_latency=0):
    temp = din 
    for i in range(extra_latency):
        temp = temp | dreg(init=0)
    
    luts = []
    for i, init_coeff in enumerate(init_coeffs):
        lut = []
        lr_err_temp = (lr_err * temp) | dreg(init=0) # shared mult
        for j in range(level):
            lut_item = Intf(din.dtype)
            pre_amp  = const(val=(j*2 + 1), tout=din.dtype) # pre-amplify for LUT
            update   = (pre_amp * lr_err_temp) | dreg(init=init_coeff*(j*2 + 1))
            lut_next = (lut_item + update) \
                | qround(fract=din.dtype.fract) | saturate(t=din.dtype)
            lut_item |= lut_next | decouple_reg(init=init_coeff, num=1)
            lut.append(lut_item)
            
        luts.append(ccat(*lut))
        if i < len(init_coeffs)-1: # not the last
            temp = temp | dreg(init=0)
    
    return ccat(*luts)


@gear
def adfe_backprop_ratio_wrap(dpred, dquant, dtarget, *, lr=0.001):
    #return adfe_backprop_ratio(dpred | dreg, dquant | dreg, dtarget | dreg, 
    #                           lr_order=lr_order) | dreg
    if dtarget != 0:
        err = dtarget - dpred
    else:
        err = dquant - dpred
    
    lr_err = const(val=lr, tout=dpred.dtype) * err \
        | qround(fract=dpred.dtype.fract) | saturate(t=dpred.dtype)
        
    coeffs = adfe_ff_adapt_coeffs(dquant, lr_err, num_tap=3, extra_latency=2)
    return coeffs
        
    
def test_coeffs():
    wl_fixp  = 16
    wl_int   = 4
    wl_fract = wl_fixp - wl_int
    
    # test seq
    len_test = 100
    preds   = [0.9, -0.9] * (100//2)
    quants  = [1, -1] * (100//2)
    targets = [1, -1] * (100//2)
    #print(xs)
    
    # setup simulation
    res = []
    reg["debug/trace"] = ['*']
    pred_drv   = drv(t=Fixp[wl_int, wl_fixp], seq=preds)
    quant_drv  = drv(t=Fixp[wl_int, wl_fixp], seq=quants)
    target_drv = drv(t=Fixp[wl_int, wl_fixp], seq=targets)
    
    adfe_backprop_ratio_wrap(pred_drv, quant_drv, target_drv, lr=0.1) \
    | collect(result=res)
    
    sim(resdir='../sim/', timeout=128, seed=1234)
    
    #print(res)
    
    for x, coeffs in zip(preds, res):
    	print(x, [float(c) for c in coeffs])
    return


def dut_hdlgen():
    # set architecture parameters
    wl_fixp = 16
    wl_int  = 4
    dtype   = Fixp[wl_int, wl_fixp]
    qtype   = Int[wl_int]
    
    dpred   = Intf(dtype)
    dquant  = Intf(dtype)
    dtarget = Intf(dtype) 
    adfe_backprop_ratio_wrap(dpred, dquant, dtarget, lr_order=5)
    hdlgen(top='/adfe_backprop_ratio_wrap', lang='sv', 
           outdir='../HDL/adfe_backprop_ratio', copy_files=True)    

    
if __name__ == '__main__':
    #test_priority_encoder()
    test_coeffs()
    #dut_hdlgen()

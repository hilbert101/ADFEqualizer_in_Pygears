import os
import numpy as np
from pygears import gear, Intf, sim, reg
from pygears.lib import dreg, decouple, const, ccat, qround, saturate
from pygears.lib import flatten, priority_mux, replicate, once, union_collapse
from pygears.lib import drv, collect
from pygears.typing import Int, Uint, Fixp, Tuple, Array, ceil_pow2, trunc, code
from pygears.hdl import hdlgen
from adfe_util import decouple_reg, mux_comb


@gear
async def priority_encoder(din: Fixp, *, lr_order) -> b'Tuple[Int[2], Uint[8]]':
    async with din as d:
        if d > 0:
            d_sgn = 1
        elif d < 0:
            d_sgn = -1
        else:
            d_sgn = 0
        d_abs = abs(d)
        
        shift = din.dtype.integer - 2
        d_max = Fixp(1 << shift)  # ignore -2^(n-1) for symmetry between + and -
        for i in range(din.dtype.width-1):
            # seems like not synthesizable
            if d_abs >= (d_max >> i):
                 d_order = Uint[8](lr_order - shift + i)
                 break
        else:
            d_order = Uint[8](0)  # din == 0
        #d_order = Uint[8](0) 
        
        yield Tuple((d_sgn, d_order))
        
        
@gear
async def priority_encoder_16(din: Fixp, *, lr_order) -> b'Tuple[Int[2], Int[8]]':
    async with din as d:
        if d > 0:
            d_sgn = 1
        elif d < 0:
            d_sgn = -1
        else:
            d_sgn = 0
        d_abs = abs(d)
        
        shift  = din.dtype.integer - 2  # ignore -2^(n-1) for symmetry between + and -
        
        # switch to Int coding (Seems like Fixp hasn't support right shift in async)
        d_code = code(d_abs << din.dtype.fract, cast_type=Int)  
        d_max  = Int(1 << (shift + din.dtype.fract))
        
        if   d_code >= (d_max >> 0):  d_order = Int[8](lr_order - shift + 0)
        elif d_code >= (d_max >> 1):  d_order = Int[8](lr_order - shift + 1)
        elif d_code >= (d_max >> 2):  d_order = Int[8](lr_order - shift + 2)
        elif d_code >= (d_max >> 3):  d_order = Int[8](lr_order - shift + 3)
        
        elif d_code >= (d_max >> 4):  d_order = Int[8](lr_order - shift + 4)
        elif d_code >= (d_max >> 5):  d_order = Int[8](lr_order - shift + 5)
        elif d_code >= (d_max >> 6):  d_order = Int[8](lr_order - shift + 6)
        elif d_code >= (d_max >> 7):  d_order = Int[8](lr_order - shift + 7)
        
        elif d_code >= (d_max >> 8):  d_order = Int[8](lr_order - shift + 8)
        elif d_code >= (d_max >> 9):  d_order = Int[8](lr_order - shift + 9)
        elif d_code >= (d_max >> 10): d_order = Int[8](lr_order - shift + 10)
        elif d_code >= (d_max >> 11): d_order = Int[8](lr_order - shift + 11)
        
        elif d_code >= (d_max >> 12): d_order = Int[8](lr_order - shift + 12)
        elif d_code >= (d_max >> 13): d_order = Int[8](lr_order - shift + 13)
        elif d_code >= (d_max >> 14): d_order = Int[8](lr_order - shift + 14)
        else:                         d_order = Int[8](lr_order - shift + 15)
        
        yield Tuple((d_sgn, d_order))
        
        """
        maximum synthesizable rate: 
            16-bit: CK = 1.2 ns
    """
             
        
@gear
def adfe_backprop_ratio(dpred, dquant, dtarget, *, lr_order):
    if dtarget != 0:
        err = dtarget - dpred
    else:
        err = dquant - dpred
    
    return priority_encoder_16(err, lr_order=lr_order)


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
        for j in range(level):
            lut_item = Intf(din.dtype)
            pre_amp  = const(val=(j*2 + 1), tout=din.dtype)
            update   = (pre_amp * lr_err * temp) | dreg(init=init_coeff*(j*2 + 1))
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
        
        



def test_priority_encoder():
    wl_fixp  = 5
    wl_int   = 2
    wl_fract = wl_fixp - wl_int
    
    # test seq
    xs = [(i - (2 ** (wl_fixp-1))) * (2 ** (-wl_fract)) for i in range(2 ** wl_fixp)]
    #print(xs)
    
    # setup simulation
    res = []
    reg["debug/trace"] = ['*']
    drv(t=Fixp[wl_int, wl_fixp], seq=xs) \
    | priority_encoder_16(lr_order=2) \
    | collect(result=res)
    
    sim(resdir='../sim/', timeout=128, seed=1234)
    
    #print(res)
    
    for x, (sgn, order) in zip(xs, res):
    	print(x, int(sgn), int(order))
    return
    
    
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

import os
import numpy as np
from pygears import gear, Intf
from pygears.lib import qround, saturate, trunc, decouple  #, dreg, 
from pygears.lib import flatten, priority_mux, replicate, once, union_collapse
from pygears.lib import const, fix, mux, ccat, when
from pygears.typing import Uint, Fixp, Tuple, Array, ceil_pow2
from adfe_util import pam4_quantizer
from adfe_util import decouple_reg as dreg


@gear
def fir_direct(din, b):
    """
        Direct DIR filter. Its order is determined by the length of b. 
        Adapted from PyGears' tutorial slides.
    """
    temp = din
    add_s = temp * b[0]
    for coef in b[1:]:
        temp = temp | dreg(init=0)
        add_s = (add_s + (temp * coef)) \
            | qround(fract=din.dtype.fract) \
            | saturate(t=din.dtype)
    return add_s 
    

@gear
async def mux_comb(ctrl: Uint, dins: Tuple) -> b'dins[0]':
    """
        Combinational MUX_N. Assume all input in dins are of the same type
    """
    async with ctrl as c:
        async with dins as ds:
            yield ds[int(c)]


@gear
def psk_quantizer(din):
    """
        Two level quantizer (modulation: phase-shift keying, PSK)
    """  
    level_pos = const(val=1.0, tout=din.dtype) #fix(din, val=1.0, tout=din.dtype) #
    level_neg = const(val=-1.0, tout=din.dtype) #fix(din, val=-1.0, tout=din.dtype) #
    #return mux(din < 0, level_pos, level_neg) | union_collapse
    return mux_comb(din < 0, ccat(level_pos, level_neg))
	
	
@gear
def prefill(din, *, val, num, dtype):
    """
        Required in all feedback loop. Adapted from PyGears' website
    """
    fill = once(val=dtype(val)) \
        | replicate(num) \
        | flatten

    return priority_mux(fill, din) \
        | union_collapse
     
        
@gear
def adaptive_coeff(din, lr_e, *, init=0.0):
    """
        Coeffecient module for adaptive filter. An example of a feedback loop:
        C(k+1) = C(k) + lr_e(k) * din(k)
        C(0) = init
    """
    delay = 1
    coeff = Intf(din.dtype)
    coeff_prev = coeff \
        | decouple(depth=ceil_pow2(delay)) \
        | prefill(val=init, num=delay, dtype=din.dtype)  # feedback part need to be like this
    #coeff_prev = coeff | dreg(init=init)  # a feedback is not valid with only dreg, 
                                           # pygears sim cannot simulate

    coeff_next = (coeff_prev + lr_e * din) \
        | qround(fract=din.dtype.fract) \
        | saturate(t=din.dtype)

    coeff |= coeff_next # connect back. must use "|=" operator
    return coeff_prev
	

@gear
def fir_adaptive(din, lr_e, *, init_coeffs=(1.0,)):
    """
        Adaptive FIR. The error term is provided externally (lr_e).
    """
    tap_num = len(init_coeffs)
    if tap_num == 0: 
        return din.dtype(0)

    temp  = din
    coeff = adaptive_coeff(temp, lr_e, init=init_coeffs[0])
    add_s = (temp * coeff) | qround(fract=din.dtype.fract) | saturate(t=din.dtype)
    for i in range(1, tap_num):
        temp  = temp | dreg(init=0)
        coeff = adaptive_coeff(temp, lr_e, init=init_coeffs[i])
        for j in range(i):
            coeff = coeff | dreg(init=0)
        add_s = (add_s + (temp * coeff)) \
            | qround(fract=din.dtype.fract) | saturate(t=din.dtype)
    return add_s


@gear
def fir_adaptive_top(din, dtarget, *, init_coeffs=(1.0,), lr=1.0, quantizer=pam4_quantizer):
    """
        Top module of adaptive FIR.
        The tap number is determined by the length of init_coeffs
        The learning rate (lr) defines the step size of coefficient adaptation.
    """
    # create variable
    lr_e = Intf(din.dtype) # learning rate times error term
    
    # feed forward
    pred = fir_adaptive(din=din, lr_e=lr_e, init_coeffs=init_coeffs)
    dout = quantizer(din=pred)
    err = (dtarget - pred) | saturate(t=din.dtype) # training
    
    lr_e_next = (err * fix(err, val=lr, tout=din.dtype)) \
        | qround(fract=din.dtype.fract) | saturate(t=din.dtype)
        
    lr_e |= lr_e_next  # connect back
	
    return ccat(dout, err)
    
    
@gear
def dfe_adaptive_top(din, dtarget, *, init_ff_coeffs=(1.0,), init_fb_coeffs=(0.0,),
                                      lr=1.0, quantizer=pam4_quantizer):
    # create variable
    lr_e = Intf(din.dtype)
    dout = Intf(din.dtype)
    fb_delay = 1
    
    # feedback decouple
    dout_prev = dout \
        | decouple(depth=ceil_pow2(fb_delay)) \
        | prefill(val=0.0, num=fb_delay, dtype=din.dtype)
    
    # feed forward and feedback
    ff   = fir_adaptive(din, lr_e=lr_e, init_coeffs=init_ff_coeffs)  # feed forward part
    fb   = fir_adaptive(dout_prev, lr_e=lr_e, init_coeffs=init_fb_coeffs)  # feedack part
    comb = (ff + fb) | saturate(t=din.dtype)
    
    # quantization and error 
    dout |= (comb | quantizer)[0]  # connect back
    ctrl = (dtarget == const(val=0.0, tout=din.dtype)) # control for training/tracking
    #dsel = mux(ctrl, dtarget | when(cond=~ctrl), dout | when(cond=ctrl)) \
    #    | union_collapse  # mux2 realization with existing gears
    dsel = mux_comb(ctrl, ccat(dtarget, dout))  # self-made gears
    
    err  = (dsel - comb) | saturate(t=din.dtype)  # error term of training/tracking
    lr_e |= (err * const(val=lr, tout=din.dtype)) \
        | qround(fract=din.dtype.fract) | saturate(t=din.dtype) # connect back
    
    return ccat(dout, err)


@gear
def dfe_adaptive_wrap(din, dtarget, *, init_ff_coeffs=(1.0,), init_fb_coeffs=(0.0,),
                                      lr=1.0, quantizer=pam4_quantizer):
    return dfe_adaptive_top(din=din | dreg(init=0.0), dtarget=dtarget  | dreg(init=0.0), \
                     init_ff_coeffs=init_ff_coeffs, init_fb_coeffs=init_fb_coeffs, \
                     lr=lr, quantizer=quantizer) | dreg

if __name__ == '__main__':
    print()
    

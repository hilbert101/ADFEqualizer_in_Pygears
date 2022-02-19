import os
import numpy as np
from pygears import gear, Intf
from pygears.lib import dreg, qround, saturate, trunc, decouple, const
from pygears.lib import flatten, priority_mux, replicate, once, union_collapse
from pygears.typing import Fixp, Tuple, Array, ceil_pow2


@gear
def fir_direct(din, b, *, fract=0):
    """
        Direct DIR filter. Its order is determined by the length of b. 
        Adapted from PyGears' tutorial slides.
    """
    tap = din
    add_s = temp * b[0]
    for coef in b[1:]:
        tap = tap | dreg(init=0)
        add_s = add_s + (tap * coef)
    return add_s | qround(fract=fract) | saturate(t=din.dtype)
    

@gear
def psk_quantizer(din):
    """
        Two level quantizer (modulation: phase-shift keying, PSK)
    """
	# don't know why, but it has to be written like this
    def level_pos():
        return const(val=din.dtype(1.0))
    def level_neg():
        return const(val=din.dtype(-1.0))
	
    if din < 0: ret = level_neg()
    else: ret = level_pos()
    return ret
	
	
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

    tap   = din
    coeff = adaptive_coeff(tap, lr_e, init=init_coeffs[0])
    add_s = tap * coeff
    for i in range(1, tap_num):
        tap   = tap | dreg(init=0)
        coeff = adaptive_coeff(tap, lr_e, init=init_coeffs[i])
        add_s = add_s + (tap * coeff)
    return add_s | qround(fract=din.dtype.fract) | saturate(t=din.dtype)


@gear
def fir_adaptive_top(din, dtarget, *, init_coeffs=(1.0,), lr=1.0, quantizer=psk_quantizer):
    """
        Top module of adaptive FIR.
        The tap number is determined by the length of init_coeffs
        The learning rate (lr) defines the step size of coefficient adaptation.
    """
    # create variable
    #zero = Fixp[din.dtype.integer, din.dtype.width](0.0)
    #lr  = Fixp[din.dtype.integer, din.dtype.width](lr)  # dont know why this doesn't work here
    #lr  = din.dtype(lr) # and this doesn't work either
    
    lr   = const(val=din.dtype(lr)) # don't use two const in one func. sim will not start
    lr_e = Intf(din.dtype) # learning rate times error term
    
    # feed forward
    dout = fir_adaptive(din=din, lr_e=lr_e, init_coeffs=init_coeffs)
    dquant = quantizer(din=dout)
    if dtarget != 0: # zero: 
        err = (dtarget - dout) | saturate(t=din.dtype) # training
    else:     
        err = (dquant - dout) | saturate(t=din.dtype) # blind tracking
    
    lr_e_tmp = lr * err | qround(fract=din.dtype.fract) | saturate(t=din.dtype)
    lr_e |= lr_e_tmp  # connect back
	
    return dout
    
    
@gear
def dfe_adaptive_top(din: Fixp, dtarget, *, init_ff_coeffs=(1.0,), init_fb_coeffs=(0.0,),
                                      lr=1.0, quantizer=psk_quantizer):
    # create variable
    lr   = const(val=din.dtype(lr)) 
    comb = Intf(din.dtype) # combined result of feed forward and feedback
    
    # quantization and error
    dout = comb #quantizer(din=comb)
    if dtarget != 0:
        err = (dtarget - comb) | saturate(t=din.dtype)  # training
    else:     
        err = (dout - comb) | saturate(t=din.dtype)  # blind tracking
    
    lr_e = (lr * err) | qround(fract=din.dtype.fract) | saturate(t=din.dtype)
    
    # decouple
    fb_delay = 1
    dout_prev = dout \
        | decouple(depth=ceil_pow2(fb_delay)) \
        | prefill(val=0.0, num=fb_delay, dtype=din.dtype)
    
    # feed forward and feedback
    ff   = fir_adaptive(din=din, lr_e=lr_e, init_coeffs=init_ff_coeffs)  # feed forward part
    fb   = fir_adaptive(din=dout_prev, lr_e=lr_e, init_coeffs=init_fb_coeffs)  # feedack part
    comb |= (ff + fb) | saturate(t=din.dtype) # connect back
    
    return dout
    

if __name__ == '__main__':
    print()
    

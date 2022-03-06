from pygears.hdl import hdlgen
from pygears.typing import Uint, Fixp, Tuple, Array, ceil_pow2
from pygears.lib import dreg, qround, saturate, decouple, pipeline, const
from pygears.lib import flatten, priority_mux, replicate, once, union_collapse
from pygears import gear, Intf

@gear
def fir_opt_retime_adaptive(din, lr_e, *, init_coeffs=(1.0,)): # -> b'din':
    tap_num = len(init_coeffs)
    if tap_num == 0: 
        return din.dtype(0)
    
    temp_prev = Intf(din.dtype)
    temp = din
    coeff = adaptive_coeff(temp, lr_e, init=init_coeffs[0])
    add_s = temp * coeff + temp_prev #first tap
    #print(add_s)
    for i in range(1, tap_num):
        add_prev2 = Intf(din.dtype)
        coeff = adaptive_coeff(temp, lr_e, init=init_coeffs[i])
        mult_delay = (temp * coeff) | dreg(init = 0)
        if i%2 == 0:
            add_prev = (mult_delay + add_prev2) \
                | qround(fract=din.dtype.fract) \
                | saturate(t=din.dtype)
            temp_prev |= dreg(add_prev)
            temp_prev = add_prev2 
        else:
            temp = dreg(temp)
            temp_prev |= (mult_delay + add_prev2) \
                | qround(fract=din.dtype.fract) |saturate(t=din.dtype) 
            temp_prev = add_prev2
    temp_prev |= const(val=0.0, tout=din.dtype)
    #print(temp_prev)
    return add_s | qround(fract=din.dtype.fract) | saturate(t=din.dtype)

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
def psk_quantizer(din):
    """
        Two level quantizer (modulation: phase-shift keying, PSK)
    """  
    level_pos = const(val=1.0, tout=din.dtype) #fix(din, val=1.0, tout=din.dtype) #
    level_neg = const(val=-1.0, tout=din.dtype) #fix(din, val=-1.0, tout=din.dtype) #
    #return mux(din < 0, level_pos, level_neg) | union_collapse
    return mux_comb(din < 0, ccat(level_pos, level_neg))

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
def fir_adaptive_top(din, dtarget, *, init_coeffs=(1.0,), lr=1.0, quantizer=psk_quantizer):
    """
        Top module of adaptive FIR.
        The tap number is determined by the length of init_coeffs
        The learning rate (lr) defines the step size of coefficient adaptation.
    """
    # create variable
    lr_e = Intf(din.dtype) # learning rate times error term
    
    # feed forward
    pred = fir_opt_retime_adaptive(din=din, lr_e=lr_e, init_coeffs=init_coeffs)
    dout = quantizer(din=pred)
    err = (dtarget - pred) | saturate(t=din.dtype) # training
    
    lr_e_next = (err * fix(err, val=lr, tout=din.dtype)) \
        | qround(fract=din.dtype.fract) | saturate(t=din.dtype)
        
    lr_e |= lr_e_next  # connect back
	
    return ccat(dout, err)

@gear
def dfe_adaptive_top(din, dtarget, *, init_ff_coeffs=(1.0,), init_fb_coeffs=(0.0,),
                                      lr=1.0, quantizer=psk_quantizer):
    # create variable
    lr_e = Intf(din.dtype)
    dout = Intf(din.dtype)
    fb_delay = 1
    
    # feedback decouple
    dout_prev = dout \
        | decouple(depth=ceil_pow2(fb_delay)) \
        | prefill(val=0.0, num=fb_delay, dtype=din.dtype)
    
    # feed forward and feedback
    ff   = fir_opt_retime_adaptive(din, lr_e=lr_e, init_coeffs=init_ff_coeffs)  # feed forward part
    fb   = fir_opt_retime_adaptive(dout_prev, lr_e=lr_e, init_coeffs=init_fb_coeffs)  # feedack part
    comb = (ff + fb) | saturate(t=din.dtype)
    
    # quantization and error 
    dout |= (comb | quantizer)  # connect back
    ctrl = (dtarget == const(val=0.0, tout=din.dtype)) # control for training/tracking
    #dsel = mux(ctrl, dtarget | when(cond=~ctrl), dout | when(cond=ctrl)) \
    #    | union_collapse  # mux2 realization with existing gears
    dsel = mux_comb(ctrl, ccat(dtarget, dout))  # self-made gears
    
    err  = (dsel - comb) | saturate(t=din.dtype)  # error term of training/tracking
    lr_e |= (err * const(val=lr, tout=din.dtype)) \
        | qround(fract=din.dtype.fract) | saturate(t=din.dtype) # connect back
    
    return ccat(dout, err)
"""
def dut_hdlgen():
    # set architecture parameters
    wl_fixp  = 16
    wl_int   = 5
    num_tap  = 8
    dtype    = Fixp[wl_int, wl_fixp]
    tup_type = Tuple[tuple([dtype] * num_tap)]
    
    din = Intf(dtype)
    b = Intf(tup_type)
    fir_opt_retime_adaptive(din,b)
 
    hdlgen(top='/fir_opt_retime', outdir='./SYN', copy_files=True, toplang='v')


if __name__ == '__main__':
    dut_hdlgen()
"""
if __name__ == '__main__':
    print()

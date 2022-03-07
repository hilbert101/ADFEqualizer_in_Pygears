from pygears.hdl import hdlgen
from pygears.typing import Uint, Fixp, Tuple
from pygears.lib import qround, saturate, decouple, pipeline, const #, dreg, 
from pygears import gear, Intf
from adfe_util import decouple_reg as dreg

"""
@gear
def fir_opt_retime(din, b): # -> b'din':
    temp_prev = Intf(din.dtype)
    temp = din 
    add_s = temp * b[0] + temp_prev #first tap
    #print(add_s)
    for i, coef in enumerate(b[1:]):
        add_prev2 = Intf(din.dtype)
        if i%2 == 0:
            mult_delay = (temp * coef) | dreg(init = 0)
            add_prev = (mult_delay + add_prev2) \
                | qround(fract=din.dtype.fract) \
                | saturate(t=din.dtype)
            temp_prev |= dreg(add_prev)
            temp_prev = add_prev2 
        else:
            mult_delay = (temp * coef) | dreg(init = 0)
            temp = dreg(temp)
            temp_prev |= (mult_delay + add_prev2) \
                | qround(fract=din.dtype.fract) |saturate(t=din.dtype) 
            temp_prev = add_prev2
    temp_prev |= const(val=0.0, tout=din.dtype)
    #print(temp_prev)
    return add_s | qround(fract=din.dtype.fract) | saturate(t=din.dtype)
"""


@gear
def fir_opt_retime(din, b):
    add_prev = Intf(din.dtype)
    temp = din 
    mult_delay = (temp * b[0]) | dreg(init = 0)  # also need dreg here
    dout = mult_delay + add_prev \
        | qround(fract=din.dtype.fract) \
        | saturate(t=din.dtype)

    for i, coef in enumerate(b[1:]):
        add = Intf(din.dtype)
        if i%2 == 0:
            mult_delay = (temp * coef) | dreg(init = 0)  # dreg must be written like this
                                                         # dreg(din) has weird bug
            add_prev |= (mult_delay + add) \
                | qround(fract=din.dtype.fract) \
                | saturate(t=din.dtype) \
                | dreg(init = 0)
        else:
            temp = temp | dreg(init = 0)
            mult_delay = (temp * coef) | dreg(init = 0)
            add_prev |= (mult_delay + add) \
                | qround(fract=din.dtype.fract) | saturate(t=din.dtype) 
        add_prev = add 
    add_prev |= const(val=0.0, tout=din.dtype)
    #print(temp_prev)
    return dout 


def dut_hdlgen():
    # set architecture parameters
    wl_fixp  = 16
    wl_int   = 5
    num_tap  = 8
    dtype    = Fixp[wl_int, wl_fixp]
    tup_type = Tuple[tuple([dtype] * num_tap)]
    
    din = Intf(dtype)
    b = Intf(tup_type)
    fir_opt_retime(din,b)
 
    hdlgen(top='/fir_opt_retime', outdir='./SYN', copy_files=True, toplang='v')


if __name__ == '__main__':
    dut_hdlgen()

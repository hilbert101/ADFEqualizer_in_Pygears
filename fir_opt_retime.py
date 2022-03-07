from pygears.hdl import hdlgen
from pygears.typing import Uint, Fixp, Tuple
from pygears.lib import qround, saturate, decouple, pipeline, const, dreg
from pygears import gear, Intf
#from adfe_util import decouple_reg as dreg


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

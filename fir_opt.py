from pygears.hdl import hdlgen
from pygears.typing import Uint, Fixp, Tuple
from pygears.lib import dreg, qround, saturate, decouple, pipeline, const
from pygears import gear, Intf

@gear
def fir_opt(din, b): # -> b'din':
    temp_prev = Intf(din.dtype)
    temp = din 
    add_s = temp * b[0] + temp_prev #first tap
    #print(add_s)
    for i, coef in enumerate(b[1:]):
        add_prev2 = Intf(din.dtype)
        if i%2 == 0:
            add_prev = (temp * coef + add_prev2) \
                | qround(fract=din.dtype.fract) \
                | saturate(t=din.dtype)
            temp_prev |= dreg(add_prev)
            temp_prev = add_prev2 
        else:
            temp = dreg(temp)
            temp_prev |= (temp * coef + add_prev2) | qround(fract=din.dtype.fract) |saturate(t=din.dtype) 
            temp_prev = add_prev2
    temp_prev |= const(val=0.0, tout=din.dtype)
    #print(temp_prev)
    return add_s | qround(fract=din.dtype.fract) | saturate(t=din.dtype)


def dut_hdlgen():
    # set architecture parameters
    wl_fixp  = 16
    wl_int   = 5
    num_tap  = 8
    dtype    = Fixp[wl_int, wl_fixp]
    tup_type = Tuple[tuple([dtype] * num_tap)]
    
    din = Intf(dtype)
    b = Intf(tup_type)
    fir_opt(din,b)
 
    hdlgen(top='/fir_opt', outdir='./syn', copy_files=True, toplang='v')


if __name__ == '__main__':
    dut_hdlgen()

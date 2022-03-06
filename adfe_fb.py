import os
import numpy as np
from pygears import gear, Intf, sim, reg
from pygears.lib import dreg, decouple, const, ccat, qround, saturate
from pygears.lib import flatten, priority_mux, replicate, once, union_collapse
from pygears.lib import drv, collect
from pygears.typing import Int, Uint, Fixp, Tuple, Array, ceil_pow2
from pygears.hdl import hdlgen
from adfe_util import decouple_reg, mux_comb, ctrl_add2, ctrl_add3, ctrl2_add3, qam16_quantizer
        
       
@gear
def adfe_fb_inner(din, b):
    comb_next = Intf(din.dtype)
    comb_prev = decouple_reg(comb_next, init=0, num=1)
    qaunt = qam16_quantizer(comb_prev)
    dout, sgn, idx = qaunt[0], qaunt[1], qaunt[2]
    
    temp = const(val=0.0, tout=din.dtype)
    for coeff in b[:-1]:
        temp = ctrl_add2(sgn, temp, coeff[idx]) | dreg(init=0)
         
    comb_next |= ctrl_add3(sgn, din, temp, b[-1][idx])
    
    return ccat(dout, comb_prev)
    
    """
        maximum synthesizable rate: 
            16-bit: 
            3-tap: CK = 1.15 ns
            4-tap: CK = 1.18 ns
            6-tap: CK = 1.25 ns
            8-tap: CK = 1.30 ns
    """
    
    
 
@gear
def adfe_fb_stag(din, b):
    dpred = Intf(din.dtype)
    dqaunt, sgn, idx = qam16_quantizer(dpred)
    
    # for the first stage, result combined with the input
    temp  = Intf(din.dtype)
    coeff = mux_comb(idx, b[0])
    dpred |= ctrl_add3(sgn, temp, din, coeff) \
        | decouple_reg(init=0, num=1)
    
    stack = [] # iterate thru all the rest coeffecients
    for lut in b[1:]:
        if len(stack) == 0:
            stack.append(lut) # do nothing
        else:
            # define current stage
            sgn_prev  = sgn | dreg(init=0)
            idx_prev  = idx | dreg(init=0)
            temp_prev = Intf(din.dtype)
            coeff_prev = mux_comb(idx_prev, lut)
            coeff      = mux_comb(idx, stack.pop())
            
            temp |= ctrl2_add3(sgn_prev, sgn, temp_prev, coeff_prev, coeff)
            
            # pass to following stage
            sgn, idx, temp = sgn_prev, idx_prev, temp_prev
    
    # for the final stage
    if len(stack) == 0:
        temp |= const(val=0.0, tout=din.dtype)
    else:
        coeff = mux_comb(idx, stack.pop())
        temp |= ctrl_add2(sgn, const(val=0.0, tout=din.dtype), coeff)
    
    return ccat(dqaunt, dpred)
    
    
        #maximum synthesizable rate: 
        #    16-bit: 
        #    8-tap: CK = 1.18 ns
    



@gear
def adfe_fb_stag_v1(din, b):
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
            temp_prev |= (temp * coef + add_prev2) \
                | qround(fract=din.dtype.fract) | saturate(t=din.dtype) 
            temp_prev = add_prev2
    temp_prev |= const(val=0.0, tout=din.dtype)
    #print(temp_prev)
    return add_s | qround(fract=din.dtype.fract) | saturate(t=din.dtype)

   
@gear
def adfe_fb_wrap(din, b):
    dout = adfe_fb_inner(din | dreg(init=0), b | dreg(init=0)) | dreg(init=0)
    
    return dout
    

def testbench():
    # generate test sequence
    test_len = 5
    #xs  = [(np.random.randint(2) * 2 - 1) for _ in range(test_len)]
    xs  = [-1, 1, -1, 1, -1]
    b   = [0.5, 0.25, 0.125]
    #b.reverse() # need to reverse indices outside
    
    # set architecture parameters
    wl_fixp  = 16
    wl_int   = 5
    num_tap  = 3
    dtype    = Fixp[wl_int, wl_fixp]
    
    # setup simulation
    res = []
    reg["debug/trace"] = [] # ['*']
    x_drv = drv(t=dtype, seq=xs)
    
    #b_drv = ccat(*[const(val=coef, tout=dtype) for coef in b])
    b_drv = ccat(*[ccat(const(val=coef, tout=dtype), 
                        const(val=3*coef, tout=dtype)) for coef in b])
    
    adfe_fb_stag(x_drv, b_drv) \
    | collect(result=res)
    sim(resdir='../sim/', timeout=10, seed=1234)
    
    for x in xs:
        print(x)
    
    for r in res:
        d, p = r
        print(float(d), float(p))
    #hdlgen(top='/fir_transpose', lang='v', outdir='../HDL', copy_files=True)
    return


def dut_hdlgen():
    # set architecture parameters
    wl_fixp  = 16
    wl_int   = 5
    num_tap  = 8
    dtype    = Fixp[wl_int, wl_fixp]
    lut_type = Tuple[dtype, dtype]  # for qam16
    tup_type = Tuple[tuple([lut_type] * num_tap)]
    
    x = Intf(dtype)
    b = Intf(tup_type)
    adfe_fb_wrap(x, b)
    hdlgen(top='/adfe_fb_wrap', lang='sv', outdir='../HDL/adfe_fb_inner', copy_files=True)
    
    
if __name__ == '__main__':
    #testbench()
    dut_hdlgen()
    

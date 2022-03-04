import os
import numpy as np
from pygears import gear, Intf, sim, reg
from pygears.lib import dreg, decouple, const, ccat
from pygears.lib import flatten, priority_mux, replicate, once, union_collapse
from pygears.lib import drv, collect
from pygears.typing import Int, Uint, Fixp, Tuple, Array, ceil_pow2, saturate
from pygears.hdl import hdlgen


@gear
def decouple_reg(din, *, init=0, num=1):
    """
        Required in all feedback loop. Adapted from PyGears' website
    """
    fill = once(val=din.dtype(init)) \
        | replicate(num) \
        | flatten

    return priority_mux(fill, din | decouple(depth=ceil_pow2(num))) \
        | union_collapse
        
        
@gear
async def psk_quantizer(din: Fixp) -> Tuple[Int[2], Uint[1]]:
    async with din as d:
        if d >= 0:
            yield Tuple[Int[2], Uint[1]]((Int[2](1), Uint[1](0)))
        else:
            yield Tuple[Int[2], Uint[1]]((Int[2](-1), Uint[1](1)))
            
            
@gear
async def qam16_quantizer(din: Fixp) -> Tuple[Int[3], Uint[1], Uint[1]]:
    dtype = Tuple[Int[3], Uint[1], Uint[1]] # quant point, sign, index
    async with din as d:
        if d >= 2:
            yield dtype((Int[3](3), Uint[1](0), Uint[1](1)))
        elif d >= 0:
            yield dtype((Int[3](1), Uint[1](0), Uint[1](0)))
        elif d >= -2:
            yield dtype((Int[3](-1), Uint[1](1), Uint[1](0)))
        else:
            yield dtype((Int[3](-3), Uint[1](1), Uint[1](1)))


@gear
async def ctrl_add2(ctrl: Uint[1], din0: Fixp, din1: Fixp) -> b'din0':
    async with ctrl as c:
        async with din0 as d0:
            async with din1 as d1:
                if c == 0:
                    yield saturate(d0 + d1, t=din0.dtype)
                else:
                    yield saturate(d0 - d1, t=din0.dtype)
                    

@gear
async def ctrl_add3(ctrl: Uint[1], din0: Fixp, din1: Fixp, din2: Fixp) -> b'din0':
    async with ctrl as c:
        async with din0 as d0:
            async with din1 as d1:
                async with din2 as d2:
                    if c == 0:
                        yield saturate(d0 + d1 + d2, t=din0.dtype)
                    else:
                        yield saturate(d0 + d1 - d2, t=din0.dtype)


@gear
async def ctrl2_add3(ctrl1: Uint[1], ctrl2: Uint[1], 
                     din0: Fixp, din1: Fixp, din2: Fixp) -> b'din0':
    async with ctrl1 as c1:
        async with ctrl2 as c2:
            async with din0 as d0:
                async with din1 as d1:
                    async with din2 as d2:
                        if c1 == 0 and c2 == 0:
                            yield saturate(d0 + d1 + d2, t=din0.dtype)
                        elif c1 == 0 and c2 == 1:
                            yield saturate(d0 + d1 - d2, t=din0.dtype)
                        elif c1 == 1 and c2 == 0:
                            yield saturate(d0 - d1 + d2, t=din0.dtype)
                        else:
                            yield saturate(d0 - d1 - d2, t=din0.dtype)
         

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
    comb_next = Intf(din.dtype)
    comb_prev = decouple_reg(comb_next, init=0, num=1)
    qaunt = qam16_quantizer(comb_prev)
    dout, sgn, idx = qaunt[0], qaunt[1], qaunt[2]
    
    # for the first stage, result combined with the input
    temp = Intf(din.dtype)
    comb_next |= ctrl_add3(sgn, temp, din, b[0][idx])
    
    stack = [] # iterate thru all the rest coeffecients
    for coeff in b[1:]:
        if len(stack) == 0:
            stack.append(coeff) # do nothing
        else:
            # define current stage
            sgn_prev  = sgn | dreg(init=0)
            idx_prev  = idx | dreg(init=0)
            temp_prev = Intf(din.dtype)
            temp |= ctrl2_add3(sgn_prev, sgn, temp_prev, coeff[idx_prev], stack.pop()[idx])
            
            # pass to following stage
            sgn, idx, temp = sgn_prev, idx_prev, temp_prev
    
    # for the final stage
    if len(stack) == 0:
        temp |= const(val=0.0, tout=din.dtype)
    else:
        temp |= ctrl_add2(sgn, const(val=0.0, tout=din.dtype), stack.pop()[idx])
    
    return ccat(dout, comb_prev)
    
    """
        maximum synthesizable rate: 
            16-bit: 
            8-tap: CK = 1.18 ns
    """
    
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
    

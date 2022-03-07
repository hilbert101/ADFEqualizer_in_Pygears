import os
import numpy as np
from pygears import gear, Intf, sim, reg
from pygears.lib import decouple, const, ccat, dreg
from pygears.lib import flatten, priority_mux, replicate, once, union_collapse
from pygears.lib import drv, collect
from pygears.typing import Int, Uint, Fixp, Tuple, Array, ceil_pow2, saturate, code
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
async def mux_comb(ctrl: Uint, dins: Tuple) -> b'dins[0]':
    """
        Combinational MUX_N. Assume all input in dins are of the same type
    """
    async with ctrl as c:
        async with dins as ds:
            yield ds[int(c)]

        
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
async def psk_quantizer(din: Fixp) -> Tuple[Int[2], Uint[1]]:
    async with din as d:
        if d >= 0:
            yield Tuple[Int[2], Uint[1]]((Int[2](1), Uint[1](0)))
        else:
            yield Tuple[Int[2], Uint[1]]((Int[2](-1), Uint[1](1)))
            
            
@gear
async def pam4_quantizer(din: Fixp) -> Tuple[b'din', Uint[1], Uint[1]]:
    dtype = Tuple[din.dtype, Uint[1], Uint[1]] # quant point, sign, index
    async with din as d:
        if d >= 2:
            yield dtype((din.dtype(3), Uint[1](0), Uint[1](1)))
        elif d >= 0:
            yield dtype((din.dtype(1), Uint[1](0), Uint[1](0)))
        elif d >= -2:
            yield dtype((din.dtype(-1), Uint[1](1), Uint[1](0)))
        else:
            yield dtype((din.dtype(-3), Uint[1](1), Uint[1](1)))
            
            
if __name__ == '__main__':
    print()

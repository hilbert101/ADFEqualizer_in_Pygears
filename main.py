import os
import numpy as np
from pygears import gear, Intf, sim, reg
from pygears.lib import dreg, qround, saturate, trunc, drv, collect, decouple, const
from pygears.lib import flatten, priority_mux, replicate, once, union_collapse
from pygears.typing import Fixp, Tuple, Array, ceil_pow2
from channel import RayleighChannel, AWGNChannel


@gear
def fir_direct(din, b, *, fract=0):
    tap = din
    add_s = temp * b[0]
    for coef in b[1:]:
        tap = tap | dreg(init=0)
        add_s = add_s + (tap * coef)
    return add_s | qround(fract=fract) | saturate(t=din.dtype)
    

@gear
def psk_quantizer(din):
	# don't know why, but it has to be written like this
    if din < 0: Fixp[din.dtype.integer, din.dtype.width](1)  
    else:       Fixp[din.dtype.integer, din.dtype.width](-1)
	
	
@gear
def prefill(din, *, val, num, dtype):
    fill = once(val=dtype(val)) \
        | replicate(num) \
        | flatten

    return priority_mux(fill, din) \
        | union_collapse
     
        
@gear
def adaptive_coeff(din, lr_e, *, init=0.0):
    delay = 1
    coeff = Intf(din.dtype)
    coeff_prev = coeff \
        | decouple(depth=ceil_pow2(delay)) \
        | prefill(val=init, num=delay, dtype=din.dtype)
    #coeff_prev = coeff | dreg(init=init)  # don't use dreg in feedback loop, 
                                           # pygears cannot simulate

    coeff_next = (coeff_prev + lr_e * din) \
        | qround(fract=din.dtype.fract) \
        | saturate(t=din.dtype)

    coeff |= coeff_next
    return coeff_prev
	

@gear
def fir_adaptive(din, lr_e, *, init_coeffs=(1.0,)):
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
def fir_adaptive_top(din, dtarget, *, init_coeffs=(1.0,), lr=1.0):	
    # create variable
    zero = Fixp[din.dtype.integer, din.dtype.width](0.0)
    #lr  = Fixp[din.dtype.integer, din.dtype.width](lr)  # dont know why this doesn't work here
    #lr  = din.dtype(lr) # and this doesn't work either
    lr   = const(val=din.dtype(lr)) # don't use two const in one func. sim will not start
    #lr_f = Fixp[din.dtype.integer, din.dtype.width](lr)
    lr_e = Intf(din.dtype) # learning rate times error term
    
    # feed forward
    dout = fir_adaptive(din=din, lr_e=lr_e, init_coeffs=init_coeffs)
    dquant = psk_quantizer(din=dout)
    if dtarget != zero: 
        err = (dtarget - dout) | saturate(t=din.dtype) # training
    else:     
        err = (dquant - dout) | saturate(t=din.dtype) # blind tracking
    
    lr_e_tmp = lr * err | qround(fract=din.dtype.fract) | saturate(t=din.dtype)
    lr_e |= lr_e_tmp  # connect back
	
    return dout
    

def fir_direct_testbench():
    # generate test sequence
    raylChan = RayleighChannel(mean_delay=50, max_delay=100, rician_factor=1.0, var_rate=0.01)
    awgnChan = AWGNChannel(pwr=0.001)
    xs_rx = [awgnChan(raylChan(np.random.randint(2) * 2 - 1)) for _ in range(100)]
    
    # set architecture parameters
    wl_fixp  = 8
    wl_int   = 3
    wl_fract = wl_fixp - wl_int
    num_tap  = 4
    init_val = tuple([1.0] + [0.0] * (num_tap-1))
    
    # filter weights
    b = Array[Fixp[wl_int, wl_fixp], num_tap](val=init_val)
    
    # setup simulation
    res = []
    reg["debug/trace"] = ['*']
    drv(t=Fixp[wl_int, wl_fixp], seq=xs_rx) \
    | fir_direct(b=b, fract=wl_fract) \
    | collect(result=res)
    sim(resdir='../sim/', timeout=128, seed=1234)
    
    for xs, r in zip(xs_rx, res):
    	print(xs, float(r))
    return
    
   
def fir_adaptive_testbench():
    # generate test sequence
    raylChan = RayleighChannel(mean_delay=1, max_delay=4, rician_factor=1.0, var_rate=0.00)
    awgnChan = AWGNChannel(pwr=0.000)
    xs_tx = []
    xs_rx = []
    
    for i in range(300):
        x_tx = np.random.randint(2) * 2.0 - 1.0
        x_rx = awgnChan(raylChan(x_tx))
        xs_tx.append(x_tx)
        xs_rx.append(x_rx)
    
    # set architecture parameters
    wl_fixp  = 8
    wl_int   = 3
    wl_fract = wl_fixp - wl_int
    num_tap  = 4
    init_coeffs = tuple([1.0] + [0.0] * (num_tap-1))
    
    # setup simulation
    res = []
    reg["debug/trace"] = ['*']
    drv_din = drv(t=Fixp[wl_int, wl_fixp], seq=xs_rx)
    drv_dtarget = drv(t=Fixp[wl_int, wl_fixp], seq=xs_tx)
    
    fir_adaptive_top(din=drv_din, dtarget=drv_dtarget, init_coeffs=init_coeffs, lr=1.0) \
    	| collect(result=res)
    #fir_adaptive_top(din=drv_din, dtarget=drv_dtarget, init_coeff=init_coeff, 
    #				 lr=1, quantizer=psk_quantizer) \
    #	| collect(result=res)
    sim(resdir='../sim/')
    
    for x_tx, x_rx, r in zip(xs_tx, xs_rx, res):
    	print(f"{x_tx:.2f}, {x_rx:.2f}, {float(r):.2f}")
    return


if __name__ == '__main__':
    # os.chdir(os.path.dirname(os.path.abspath(__file__)))
    # print(os.getcwd())
    # fir_direct_testbench()
    fir_adaptive_testbench()

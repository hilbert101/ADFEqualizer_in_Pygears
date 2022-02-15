import numpy as np
from matplotlib import pyplot as plt
from channel import RayleighChannel, AWGNChannel


class AdaptiveDecisionFeedbackEqualizer:
    def __init__(self, fftap=12, fbtap=8, lr=0.001, qfunc=lambda x: np.sign(x)):
        self.fb_coeffs = np.zeros(fbtap)
        self.ff_coeffs = np.zeros(fftap)
        self.ff_coeffs[0] = 1.0
        self.ff_mems = np.zeros(fftap)
        self.fb_mems = np.zeros(fbtap)
        self.lr = lr
        self.qfunc = qfunc
        return

    def __call__(self, x, d_target=None):
        self.ff_mems = np.roll(self.ff_mems, 1)
        self.ff_mems[0] = x
        y = np.sum(self.ff_coeffs * self.ff_mems)
        z = 0
        if len(self.fb_mems) > 0:
            z = np.sum(self.fb_coeffs * self.fb_mems)
        v = y + z
        d = d_out = self.qfunc(v)  # blind tracking
        if d_target is not None: d = d_target  # training
        e = d - v
        self.ff_coeffs += self.lr * e * self.ff_mems
        self.fb_coeffs += self.lr * e * self.fb_mems
        if len(self.fb_mems) > 0:
            self.fb_mems = np.roll(self.fb_mems, 1)
            self.fb_mems[0] = d

        return d_out, e


def test_adfe():
    raylChan = RayleighChannel(mean_delay=3, max_delay=50, rician_factor=1.0, var_rate=0.0)
    awgnChan = AWGNChannel(pwr=0.001)
    qfunc = lambda x: np.sign(x)  # PSK quantization, 2 quantization level
    adfe = AdaptiveDecisionFeedbackEqualizer(fftap=12, fbtap=8, lr=0.03, qfunc=qfunc)
    ts = []
    es = []

    for i in range(3000):
        x_tx = np.random.randint(2) * 2 - 1
        x_rx = awgnChan(raylChan(x_tx))
        d_rx, e = adfe(x_rx, d_target=x_tx)
        # evm = np.abs(d_rx - x_tx)
        ts.append(i)
        es.append(np.abs(e))

    print(raylChan.get_response())
    print(adfe.ff_coeffs)
    print(adfe.fb_coeffs)
    plt.plot(ts, es)
    plt.show()
    return


if __name__ == '__main__':
    test_adfe()
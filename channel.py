import numpy as np
from matplotlib import pyplot as plt


class Channel:
    """
    Generic channel, do nothing to the input signal
    """
    def __init__(self):
        return

    def __call__(self, x):
        return x


class AWGNChannel(Channel):
    """
    Additive Gaussian white noise
    """
    def __init__(self, pwr=1.0):
        """
            pwr: average power of Gaussian white noise
        """
        super().__init__()
        self.stddev = np.sqrt(pwr)
        return

    def __call__(self, x):
        return x + np.random.normal(0, self.stddev)


class RayleighChannel(Channel):
    """
    Frequency Selective Multi-path Rayleigh Fading channel, not include Doppler effect
    See https://www.gaussianwaves.com/2020/08/rician-flat-fading-channel-simulation/
        https://www.gaussianwaves.com/2016/10/modeling-a-frequency-selective-multipath-fading-channel-using-tdl-filters/
        https://www.gaussianwaves.com/2010/02/fading-channels-rayleigh-fading-2/
        https://www.gaussianwaves.com/2014/07/power-delay-profile/
        http://www.idc-online.com/technical_references/pdfs/electrical_engineering/Power_Delay_Profile.pdf
    """
    def __init__(self, mean_delay=50, max_delay=100, rician_factor=1.0, var_rate=0.0):
        """
            mean_delay: mean delay in power delay profile. larger means more ISI
            max_delay: max delay in power delay profile. larger means more ISI
            rician_factor: ratio of directly transmited and scattered signal in power.
                           0.0 < rician_factor < infinity. larger means cleaner rx signal.
            var_rate: variation rate of the channel's response. 
        """
        super().__init__()
        # gain is set to ~1
        self.size = max_delay + 1

        delay_pwr_decay = np.exp(-1/mean_delay)
        scatter_pwr = 1 / (1 + rician_factor)
        los_pwr = 1 - scatter_pwr
        scatter_pwr_0 = scatter_pwr * (1 - delay_pwr_decay) / (1 - delay_pwr_decay ** self.size)
        pwr_0 = los_pwr + scatter_pwr_0
        k_factor = los_pwr / scatter_pwr_0

        self.mean = np.sqrt(pwr_0 * k_factor / (2 * (k_factor + 1)))
        self.stddevs = np.sqrt([pwr_0 / (2 * (k_factor + 1))] +
                               [scatter_pwr_0 * (delay_pwr_decay ** p) / 2 for p in range(1, self.size)])

        self.decays = 1 - 0.5 * np.square(var_rate / self.stddevs)
        self.decays = np.clip(self.decays, 0.0, 1.0)
        self.stepdevs = np.sqrt(1 - np.square(self.decays)) * self.stddevs
        self.gauss = self.stddevs * np.random.normal(0, 1, (2, self.size))  # complex coeffs = mean (0) + gauss (0~size)
        self.mems = np.zeros(self.size)
        return

    def __call__(self, x):
        self.mems = np.roll(self.mems, 1)
        self.mems[0] = x

        return np.sum(self.get_response() * self.mems)

    def get_response(self):
        h_0 = self.mean + self.gauss[:, 0]
        h_mag_0 = np.sqrt(np.sum(np.square(h_0)))
        h_norm_0 = h_0 / h_mag_0
        h_mag = np.matmul(h_norm_0, self.gauss)
        h_mag[0] = h_mag_0

        return h_mag

    def vary(self):
        steps = self.stepdevs * np.random.normal(0, 1, (2, self.size))
        self.gauss = self.decays * self.gauss + steps
        return


def test_channel_response():
    raylChan = RayleighChannel(mean_delay=50, max_delay=100, rician_factor=1.0, var_rate=0.01)
    ts = []
    hs = []
    for i in range(10000):
        h = raylChan.get_response()
        h_mag = np.sqrt(np.sum(np.square(h)))
        # print(h_mag)
        raylChan.vary()
        ts.append(i)
        hs.append(h_mag)

    plt.plot(ts, hs)
    plt.show()
    return


if __name__ == '__main__':
    test_channel_response()
    

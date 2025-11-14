import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as sg
import sounddevice as sd

if __name__ == "__main__":
    fs = 48000

    # Part one -- simulate a simple channel
    # path_delay = np.array([0, 3, 20, 15, 2, 4, 8]) / 1e3
    # path_gain = np.array([1, -0.4, -0.5, 0.6, 0.3, 0.2, -0.5])
    # x = sg.chirp(np.arange(0, 1*fs)/fs, 100, 1, fs/2)
    # x = np.tile(x, 3)
    # x = np.concatenate((np.zeros(fs,), x, np.zeros(fs,)))
    # y = np.zeros(len(x))
    # for n in range(3):
    #     y = y + np.roll(path_gain[n] * x, np.round(path_delay[n]*fs))
    # plt.figure()
    # plt.plot(y)
    # plt.show()

    # Part two -- play a chirp signal through the real system
    duration = 5
    chirp = sg.chirp(np.arange(0, duration*fs)/fs, 100, duration, fs/2, method='log')
    x = np.tile(chirp, 1)
    x = np.concatenate((np.zeros(fs,), x, np.zeros(fs,)))

    y = sd.playrec(x, fs, channels=1, blocking=True)[:, 0]  # left channel only

    plt.figure(figsize=(15, 3))

    plt.subplot(151)
    plt.plot(x)
    plt.title('x-time')

    plt.subplot(152)
    plt.specgram(x, Fs=fs)
    plt.title('x-spectrogram')

    plt.subplot(153)
    plt.plot(y)
    plt.title('y-time')

    plt.subplot(154)
    plt.specgram(y, Fs=fs)
    plt.title("y-spectrogram")

    plt.subplot(155)
    xcorr = sg.fftconvolve(y, chirp[::-1])
    plt.plot(np.arange(len(xcorr))/fs, xcorr)
    plt.title("cross-correlation")

    plt.tight_layout()
    plt.show()

     
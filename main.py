import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import chirp
from scipy import signal


def simulate_chirp_response(t,delay,gain,input_signal,SAMPLING_FREQUENCY):

    output_signal = np.zeros(len(t) + np.round(int (delay[-1]*SAMPLING_FREQUENCY )) +1 ) # the output sig shouldnt be longer than the latest delay + orig sig length
                                                # ^^^^ POTENTIAL ROUNDING ISSUE WITH INDEXING
    for index in range(len(delay)):
        delay_index = int(np.round(delay[index] * SAMPLING_FREQUENCY))
        gain_factor = gain[index]

        start_index = delay_index
        end_index = min(delay_index + len(input_signal), len(output_signal)) # my decl for output_signal prevents weirdness in indexing, this is just a safeguard
        output_signal[start_index:end_index] += input_signal[:end_index-start_index] * gain_factor
    return output_signal
    

    # assumptions: delay and gain are same length
def simulate_chirp_response_convolution(delay, gain, input_signal, SAMPLING_FREQUENCY):
    h_len = int(np.round(delay[-1] * SAMPLING_FREQUENCY)) + 1
    h = np.zeros(h_len)
    for idx in range(len(delay)):
        h[int(np.round(delay[idx] * SAMPLING_FREQUENCY))] = gain[idx]
    output_signal = np.convolve(input_signal, h)
    return output_signal

def create_spectrogram(title,sampling_freq, data, ax=None):
    # Create spectrogram on the provided axes or a new one
    if ax is None:
        fig, ax = plt.subplots(figsize=(18, 5))
    Pxx, freqs, bins, im = ax.specgram(data, Fs=sampling_freq, cmap="rainbow")
    
    ax.set_title(title)
    ax.set_xlabel('Time [s]')
    ax.set_ylabel('Frequency [Hz]')
    return ax  # return the axes object

def chirp_and_impulse_practice_functions(chirp_signal, t, SAMPLING_FREQUENCY):

    d = np.array([0, 0.003, 0.010])  # delays in seconds
    g = np.array([1, -0.7, 0.5])     # gains

    # Compute responses
    o = simulate_chirp_response(t, d, g, chirp_signal, SAMPLING_FREQUENCY)
    o2 = simulate_chirp_response_convolution(d, g, chirp_signal, SAMPLING_FREQUENCY)

    dt = t[1] - t[0]         
    extended_t = np.arange(len(o)) * dt

    plot_samples = 2000  # number of samples to plot

    fig, axs = plt.subplots(1, 6, figsize=(30, 6))

    # 1. Original chirp

    plt.title('x-spectrogram')
    axs[0].specgram(chirp_signal, Fs=SAMPLING_FREQUENCY)
    axs[0].set_title('Chirp Signal')
    axs[0].set_xlabel('Time (s)')
    axs[0].set_ylabel('Amplitude')
    axs[0].grid(True)

    # 2. Simulated response (first 2000 samples)
    axs[1].plot(extended_t[:plot_samples], o[:plot_samples], marker='o', linestyle='-', markersize=2)
    axs[1].set_title("simulate_chirp_response")
    axs[1].set_xlabel("Time (s)")
    axs[1].set_ylabel("Amplitude")
    axs[1].grid(True)

    # 3. Convolution response (first 2000 samples)
    axs[2].plot(extended_t[:plot_samples], o2[:plot_samples], marker='o', linestyle='-', markersize=2)
    axs[2].set_title("simulate_chirp_response_convolution")
    axs[2].set_xlabel("Time (s)")
    axs[2].set_ylabel("Amplitude")
    axs[2].grid(True)

    # 4. Overlay comparison (first 2000 samples)
    axs[3].plot(extended_t[:plot_samples], o[:plot_samples], label='Direct Response')
    axs[3].plot(extended_t[:plot_samples], o2[:plot_samples], '--', label='Convolution')
    axs[3].set_title("Overlay Comparison")
    axs[3].set_xlabel("Time (s)")
    axs[3].set_ylabel("Amplitude")
    axs[3].legend(loc='upper right', bbox_to_anchor=(0.5, -0.3), ncol=2,fontsize = 'small')

    axs[3].grid(True)

    # 5 & 6. Spectrograms (full signals)
    create_spectrogram("Chirp Response by Indexing",SAMPLING_FREQUENCY, o, axs[4])
    create_spectrogram("Chirp Response by Convolution",SAMPLING_FREQUENCY, o2, axs[5])

    plt.tight_layout()
    return fig

def main():
    
    START = 0
    STOP = 10
    NUM_SAMPLES = 480000
    t = np.linspace(START, STOP, NUM_SAMPLES)

    SAMPLING_INTERVAL = (STOP - START) / (NUM_SAMPLES - 1)
    SAMPLING_FREQUENCY = 1 / SAMPLING_INTERVAL


    # Chirp
    chirp_end_freq = 1000  # desired max frequency
    NYQUIST_FREQUENCY = SAMPLING_FREQUENCY / 2
    max_chirp_frequency = min(chirp_end_freq, NYQUIST_FREQUENCY)
    y = chirp(t, f0=0, f1=max_chirp_frequency, t1=STOP, method="linear")


    fig = chirp_and_impulse_practice_functions(y, t, SAMPLING_FREQUENCY)
    plt.show()


print("SOP")
main()
print("EOP")





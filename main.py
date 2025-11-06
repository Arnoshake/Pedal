import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import chirp
from scipy import signal


def simulate_chirp_response(t,delay,gain,input_signal,SAMPLING_FREQUENCY):

    output_signal = np.zeros(len(t) + np.round(int (delay[-1]*SAMPLING_FREQUENCY )) ) # the output sig shouldnt be longer than the latest delay + orig sig length
                                                # ^^^^ POTENTIAL ROUNDING ISSUE WITH INDEXING
    for index in range(len(delay)):
        delay_index = int(np.round(delay[index] * SAMPLING_FREQUENCY))
        gain_factor = gain[index]

        start_index = delay_index
        end_index = min(delay_index + len(input_signal), len(output_signal)) # my decl for output_signal prevents weirdness in indexing, this is just a safeguard
        output_signal[start_index:end_index] += input_signal[:end_index-start_index] * gain_factor
    return output_signal
    # delay and gain belong to h(t)
    # signal is the input
    # using the signal, apply the delay and gain and find the new y(t)
    # THis function is the first step to eventually assuming/determining h(t) from a given x(t) and y(t)

    # assumptions: delay and gain are same length
def simulate_chirp_response_convolution(delay,gain,input_signal,SAMPLING_FREQUENCY):
 
    h = np.zeros( int(np.round(delay[-1]*SAMPLING_FREQUENCY) ) )
    for index in range(len(delay)):
        #assign the gain value of index to the spot of the delay
        x = int(np.floor(delay[index] * SAMPLING_FREQUENCY))
        y = gain[index]
        h[x] = y
    output_signal = np.convolve(input_signal, h)
    return output_signal
def create_spectrogram(sampling_freq, data, ax=None):
    # Create spectrogram on the provided axes or a new one
    if ax is None:
        fig, ax = plt.subplots(figsize=(18, 5))
    Pxx, freqs, bins, im = ax.specgram(data, Fs=sampling_freq, cmap="rainbow")
    
    ax.set_title('Spectrogram')
    ax.set_xlabel('Time [s]')
    ax.set_ylabel('Frequency [Hz]')
    # plt.plot(np.arange(len(h))/SAMPLING_FREQUENCY, h) ,--- READ!
    return ax  # return the axes object

def chirp_and_impulse_practice_functions(chirp_signal,t,SAMPLING_FREQUENCY):

    d = np.array([0,0.003,0.010]) #passing in time in terms of seconds!
    g = np.array([1,-0.7,0.5])
    
    o =simulate_chirp_response(t,d,g,chirp_signal,SAMPLING_FREQUENCY)
    o2 = simulate_chirp_response_convolution(d,g,chirp_signal,SAMPLING_FREQUENCY)

    extended_t = np.linspace(0, len(o)/len(t)*t[-1], len(o))
    fig, axs = plt.subplots(1, 6, figsize=(18, 4))

    axs[0].plot(t,chirp_signal)
    axs[0].set_title('Chirp Signal')
    axs[0].set_xlabel('Time (t)')
    axs[0].set_ylabel('Amplitude')
    axs[0].grid(True)

    axs[1].plot(extended_t[:50], o[:50], marker='o', linestyle='-', markersize=4)
    axs[1].set_title("simulate_chirp_response")
    axs[1].set_xlabel("Time (s)")
    axs[1].set_ylabel("Amplitude")
    axs[1].grid(True)

    axs[2].plot(extended_t[:len(o2)], o2, marker='o', linestyle='-', markersize=4)
    axs[2].set_title("simulate_chirp_response_convolution")
    axs[2].set_xlabel("Time (s)")
    axs[2].set_ylabel("Amplitude")
    axs[2].grid(True)

    create_spectrogram(SAMPLING_FREQUENCY,o,axs[4])
    create_spectrogram(SAMPLING_FREQUENCY,o2,axs[5])
    

    plt.tight_layout()  # makes sure labels/titles don’t overlap
    return fig,axs[3]

def main():
    START = 0
    STOP = 10
    NUM_SAMPLES = 480000
    t = np.linspace(START,STOP,NUM_SAMPLES) # start: 0, stop: 10, increments: 1000x --> sampling interval = (10-0)/(1000-1) = 0.01001 seconds
    
    SAMPLING_INTERVAL = (STOP - START)/(NUM_SAMPLES - 1) # seconds per sample
    SAMPLING_FREQUENCY = 1/ SAMPLING_INTERVAL # samples per second
    # sampling rate = samples/second = 1 / 0.01001 = 99.9Hz

    NYQUIST_FREQUENCY = SAMPLING_FREQUENCY / 2
   
    
    chirp_end_freq = 1000 #WHAT USER WANTS 
    max_chirp_frequency = min(chirp_end_freq,NYQUIST_FREQUENCY) #ensures aliasing does not occur
    y = chirp(t,f0=0,f1=max_chirp_frequency,t1=10,method="linear") 
        # your max frequency of chirp can only be 1/2 of your sampling frequency --> NYQUIST THEOREM
  
    results, ax3 = chirp_and_impulse_practice_functions(y,t,SAMPLING_FREQUENCY)
    spectro_fig = create_spectrogram(SAMPLING_FREQUENCY,y,ax3)

    plt.figure(results)
    plt.show()

print("SOP")
main()
print("EOP")





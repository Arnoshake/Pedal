import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import chirp
from scipy import signal

def time_to_index(time, t_array): #GPT helper function to convert a time to an index for np.arrays
    dt = t_array[1] - t_array[0]
    return int(time / dt)
def simulate_chirp_response(linSpace, delay,gain,signal):
    
    output_signal = np.zeros(len(linSpace) + time_to_index(delay[-1],linSpace)) # the output signal shouldnt be longer than the latest delay + original signal length
    
    
    for index, val in enumerate(delay):
        delay_index = time_to_index(delay[index],linSpace)
        gain_factor = gain[index]

        start_index = delay_index
        end_index = min(delay_index + len(signal), len(output_signal)) # my decl for output_signal prevents weirdness in indexing, this is just a safeguard
        output_signal[start_index:end_index] += signal[:end_index-start_index] * gain_factor
    return output_signal
    # delay and gain belong to h(t)
    # signal is the input
    # using the signal, apply the delay and gain and find the new y(t)
    # THis function is the first step to eventually assuming/determining h(t) from a given x(t) and y(t)

    # assumptions: delay and gain are same length
def simulate_chirp_response_convolution(input_linSpace, delay,gain,signal):
    #assign the gain value of index to the spot of the delay
    h = np.zeros( time_to_index(delay[-1],input_linSpace) + 1)
    for index,val in enumerate(delay):
        x = time_to_index(delay[index],input_linSpace)
        y = gain[index]
        h[x] = y

    # need to recreate h by combining the data from delay and gain
    output_signal = np.convolve(signal, h)
    return output_signal
def practice_functions(chirp_signal,t):
    x = np.array([1,2,3,4,5])
    d = np.array([0,0.003,0.010]) #passing in time in terms of seconds!
    g = np.array([1,-0.7,0.5])
    o =simulate_chirp_response(t,d,g,x)
    o2 = simulate_chirp_response_convolution(t,d,g,x)

    extended_t = np.linspace(0, len(o)/len(t)*t[-1], len(o))
    fig, axs = plt.subplots(1, 4, figsize=(12, 4))

    axs[0].plot(t,chirp_signal)
    axs[0].set_title('Chirp Signal')
    axs[0].set_xlabel('Time (t)')
    axs[0].set_ylabel('Amplitude')
    axs[0].grid(True)

    axs[1].plot(extended_t[:50], o[:50], marker='o', linestyle='-', markersize=4)
    axs[1].set_title("Zoomed-In Chirp Response")
    axs[1].set_xlabel("Time (s)")
    axs[1].set_ylabel("Amplitude")
    axs[1].grid(True)

    axs[2].plot(extended_t[:len(o2)], o2, marker='o', linestyle='-', markersize=4)
    axs[2].set_title("Zoomed-In Chirp Response Convolution")
    axs[2].set_xlabel("Time (s)")
    axs[2].set_ylabel("Amplitude")
    axs[2].grid(True)
    

    plt.tight_layout()  # makes sure labels/titles don’t overlap
    return fig,axs[3]
def create_spectrogram(sampling_freq, data, ax=None):
    # Create spectrogram on the provided axes or a new one
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 5))
    Pxx, freqs, bins, im = ax.specgram(data, Fs=sampling_freq, cmap="rainbow")
    
    ax.set_title('Spectrogram')
    ax.set_xlabel('Time [s]')
    ax.set_ylabel('Frequency [Hz]')
    
    return ax  # return the axes object

def main():
    START = 0
    STOP = 10
    NUM_SAMPLES = 10000
    t = np.linspace(START,STOP,NUM_SAMPLES) # start: 0, stop: 10, increments: 1000x --> sampling interval = (10-0)/(1000-1) = 0.01001 seconds
    
    SAMPLING_INTERVAL = (STOP - START)/(NUM_SAMPLES - 1)
    SAMPLING_FREQUENCY = 1/ SAMPLING_INTERVAL
    # sampling rate = samples/second = 1 / 0.01001 = 99.9Hz

    NYQUIST_FREQUENCY = SAMPLING_FREQUENCY / 2
    
    chirp_end_freq = 1000 #WHAT USER WANTS 
    max_chirp_frequency = min(chirp_end_freq,NYQUIST_FREQUENCY) #ensures aliasing does not occur
    y = chirp(t,f0=0,f1=max_chirp_frequency,t1=10,method="linear") # your max frequency of chirp can only be 1/2 of your sampling frequency --> NYQUIST THEOREM
        # chirp is applied across the linspace of t, the frequency spans 6Hz to 1Hz
        # it goes from 0 until t1
  
    results, ax3 = practice_functions(y,t)
    spectro_fig = create_spectrogram(SAMPLING_FREQUENCY,y,ax3)

    plt.figure(results)
    plt.show()

print("SOP")
main()
print("EOP")





# WHEN YOU HAVE LIBRARY IMPORTS, YOU STILL HAVE TO MAKE SURE ITS DOWNLOADED TO YOUR COMPTUER
# pip3 install NAME 
# NAME = numpy,matplotlib,scipy,scipy

import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import chirp
from scipy import signal

def time_to_index(time, t_array): #GPT helper function to convert a time to an index for np.arrays
    dt = t_array[1] - t_array[0]
    return int(time / dt)
def simulate_chirp_response(linSpace, delay,gain,signal):
    # CREATE THE CHIRP RESPONSE MANUALLY
    # MEANING THAT YOU CREATE AN OUTPUT Np.Array THAT HOLDS THE SUM OF THE EFFECTS OF ALL SIGNALS
    dummy = 0
def simulate_chirp_response_convolution(input_linSpace, delay,gain,signal):
   dummy = 0
def practice_functions(t):
    x = np.array([1,2,3,4,5])
    
    d = np.array([0,0.003,0.010]) #passing in time in terms of seconds!
    g = np.array([1,-0.7,0.5])
    o =simulate_chirp_response(t,d,g,x)
    o2 = simulate_chirp_response_convolution(t,d,g,x)

    extended_t = np.linspace(0, len(o)/len(t)*t[-1], len(o))
    fig, axs = plt.subplots(1, 2, figsize=(12, 4))

    axs[0].plot(extended_t[:50], o[:50], marker='o', linestyle='-', markersize=4)
    axs[0].set_title("Zoomed-In Chirp Response")
    axs[0].set_xlabel("Time (s)")
    axs[0].set_ylabel("Amplitude")
    axs[0].grid(True)

    axs[1].plot(extended_t[:len(o2)], o2, marker='o', linestyle='-', markersize=4)
    axs[1].set_title("Zoomed-In Chirp Response Convolution")
    axs[1].set_xlabel("Time (s)")
    axs[1].set_ylabel("Amplitude")
    axs[1].grid(True)

    plt.tight_layout()  # makes sure labels/titles don’t overlap
    plt.show()

def main():
    t = np.linspace(0,10,10000) # start: 0, stop: 10, increments: 1000x --> sampling interval = (10-0)/(1000-1) = 0.01001 seconds
    # sampling rate = samples/second = 1 / 0.01001 = 99.9Hz

    y = chirp(t,f0=6,f1=1,t1=10,method="linear")
        # chirp is applied across the linspace of t, the frequency spans 6Hz to 1Hz
        # it goes from 0 until t1
        # the chirp is linear

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(t,y)

    ax.set_title('Chirp Signal')
    ax.set_xlabel('Time (t)')
    ax.set_ylabel('Amplitude')
    ax.grid(True)

    practice_functions(t)

print("SOP")
main()
print("EOP")

list_a = [1,2,3,4,5]





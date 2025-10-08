import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import chirp
from scipy import signal

t = np.linspace(0,10,1000) # start: 0, stop: 10, increments: 1000x

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

# Display the plot
plt.show()

import matplotlib.pyplot as plt
import numpy as np

# Stepsizes (Epsilon)
stepsizes = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])

# ASR for various mismatch lengths
asr_single = np.array([42.00, 60.00, 74.00, 84.00, 87.00, 97.00, 98.00, 93.00, 96.00])
asr_quarter = np.array([1.00, 10.00, 16.00, 25.00, 33.00, 38.00, 46.00, 55.00, 56.00])
asr_half = np.array([0.00, 0.00, 2.00, 3.00, ])
asr_threequarter = np.array([0.00, 0.00])

# Plot ASR vs Stepsize
plt.plot(stepsizes, asr_single, label='>= 1 Bit Mismatch')
plt.plot(stepsizes, asr_quarter, label='>= 8 Bit Mismatch')
plt.title('Attack Success Rate (ASR) vs Stepsize')
plt.xlabel('Stepsize')
plt.ylabel('ASR (%)')
plt.legend()
plt.grid()
plt.show()

# L2 distortion for various mismatch lengths
l2_single = np.array([3.57, 5.89, 8.20, 9.92, 11.11, 12.07, 13.23, 13.20, 12.63])
l2_quarter = np.array([5.76, 10.48, 14.68, 15.28, 22.16, 24.47, 28.15, 27.30, 28.27])
l2_half = np.array([0.00, 0.00, 18.42, 24.53, ])
l2_threequarter = np.array([0.00, 0.00])

# Plot L2 Distortion vs Stepsize
plt.plot(stepsizes, l2_single, label='>= 1 Bit Mismatch')
plt.plot(stepsizes, l2_quarter, label='>= 8 Bit Mismatch')
plt.title('L2 Distortion vs Stepsize')
plt.xlabel('Stepsize')
plt.ylabel('L2 Distortion')
plt.legend()
plt.grid()
plt.show()

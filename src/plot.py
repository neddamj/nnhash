import matplotlib.pyplot as plt
import numpy as np

# Stepsizes (Epsilon)
stepsizes = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])

# ASR for various mismatch lengths
asr_single = np.array([42.00, 60.00, 74.00, 84.00, 87.00, 97.00, 98.00, 93.00, 96.00])
asr_quarter = np.array([])
asr_half = np.array([])
asr_threequarter = np.array([])

# Plot ASR vs Stepsize
plt.plot(stepsizes, asr_single, label='>1 Bit Mismatch')
plt.title('Attack Success Rate (ASR) vs Stepsize')
plt.xlabel('Stepsize')
plt.ylabel('ASR (%)')
plt.legend()
plt.grid()
plt.show()

# L2 distortion for various mismatch lengths
l2_single = np.array([3.57, 5.89, 8.20, 9.92, 11.11, 12.07, 13.23, 13.20, 12.63])
l2_quarter = np.array([])
l2_half = np.array([])
l2_threequarter = np.array([])

# Plot L2 Distortion vs Stepsize
plt.plot(stepsizes, l2_single, label='>1 Bit Mismatch')
plt.title('L2 Distortion vs Stepsize')
plt.xlabel('Stepsize')
plt.ylabel('L2 Distortion')
plt.legend()
plt.grid()
plt.show()

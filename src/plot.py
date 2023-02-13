import matplotlib.pyplot as plt
import numpy as np

"""# Stepsizes (Epsilon)
stepsizes = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])

# ASR for various mismatch lengths - 5000 steps
asr_single = np.array([42.00, 60.00, 74.00, 84.00, 87.00, 97.00, 98.00, 93.00, 96.00])
asr_quarter = np.array([1.00, 10.00, 16.00, 25.00, 33.00, 38.00, 46.00, 55.00, 56.00])
asr_half = np.array([0.00, 0.00, 2.00, 3.00, 3.00, 4.00, 4.00, 5.00, 7.00])
asr_threequarter = np.array([0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00])
asr_threequarter_10k = np.array([0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 3.00, 5.00])

# Plot ASR vs Stepsize
plt.plot(stepsizes, asr_single, label='>= 1 Bit Mismatch')
plt.plot(stepsizes, asr_quarter, label='>= 8 Bit Mismatch')
plt.plot(stepsizes, asr_half, label='>= 16 Bit Mismatch')
plt.plot(stepsizes, asr_threequarter, label='>= 24 Bit Mismatch - 5k Queries')
plt.title('Attack Success Rate (ASR) vs Stepsize')
plt.xlabel('Stepsize')
plt.ylabel('ASR (%)')
plt.legend()
plt.grid()
plt.show()

# L2 distortion for various mismatch lengths - 5000 steps
l2_single = np.array([3.57, 5.89, 8.20, 9.92, 11.11, 12.07, 13.23, 13.20, 12.63])
l2_quarter = np.array([5.76, 10.48, 14.68, 15.28, 22.16, 24.47, 28.15, 27.30, 28.27])
l2_half = np.array([0.00, 0.00, 18.42, 24.53, 27.95, 31.27, 26.34, 30.91, 33.18])
l2_threequarter = np.array([0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00])
l2_threequarter_10k = np.array([0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 48.67, 60.61])

# Plot L2 Distortion vs Stepsize
plt.plot(stepsizes, l2_single, label='>= 1 Bit Mismatch')
plt.plot(stepsizes, l2_quarter, label='>= 8 Bit Mismatch')
plt.plot(stepsizes, l2_half, label='>= 16 Bit Mismatch')
plt.plot(stepsizes, l2_threequarter, label='>= 24 Bit Mismatch - 5k Queries')
plt.title('L2 Distortion vs Stepsize')
plt.xlabel('Stepsize')
plt.ylabel('L2 Distortion')
plt.legend()
plt.grid()
plt.show()"""

# Queries
queries = np.array([500, 1000, 1500, 2000, 2500, 3000, 3500, 4000, 4500, 5000])

# ASR at 1 mismatched bit threshold
asr_one_1 = np.array([6, 8, 10, 11, 13, 15, 16, 17, 19, 21])
asr_one_2 = np.array([27, 30, 34, 38, 43, 46, 54, 57, 60, 60])
asr_one_3 = np.array([29, 37, 48, 54, 56, 59, 62, 65, 70, 74])
asr_one_4 = np.array([36, 50, 62, 67, 73, 77, 77, 78, 83, 84])
asr_one_5 = np.array([46, 59, 68, 73, 78, 79, 82, 86, 86, 87])
asr_one_6 = np.array([48, 58, 73, 84, 89, 92, 94, 94, 97, 97])
asr_one_7 = np.array([56, 72, 79, 82, 87, 92, 96, 97, 98, 98])
asr_one_8 = np.array([51, 71, 79, 84, 88, 89, 89, 89, 92, 93])
asr_one_9 = np.array([54, 71, 80, 85, 89, 90, 90, 92, 94, 96])

# Plot ASR (1 bit mismatch) vs queries
plt.plot(queries, asr_one_1, label='Stepsize: 0.1')
plt.plot(queries, asr_one_2, label='Stepsize: 0.2')
plt.plot(queries, asr_one_3, label='Stepsize: 0.3')
plt.plot(queries, asr_one_4, label='Stepsize: 0.4')
plt.plot(queries, asr_one_5, label='Stepsize: 0.5')
plt.plot(queries, asr_one_6, label='Stepsize: 0.6')
plt.plot(queries, asr_one_7, label='Stepsize: 0.7')
plt.plot(queries, asr_one_8, label='Stepsize: 0.8')
plt.plot(queries, asr_one_9, label='Stepsize: 0.9')
plt.title('ASR vs Queries (1  bit mismatch)')
plt.xlabel('Queries')
plt.ylabel('ASR (%)')
plt.legend()
plt.grid()
plt.show()

# ASR at 8 mismatched bit threshold
asr_eight_1 = np.array([0, 0, 0, 0, 0, 0, 1, 1, 1, 1])
asr_eight_2 = np.array([0, 0, 1, 2, 2, 6, 6, 7, 7, 7])
asr_eight_3 = np.array([0, 3, 3, 4, 6, 10, 11, 13, 16, 16])
asr_eight_4 = np.array([2, 5, 10, 12, 16, 18, 21, 24, 24, 25])
asr_eight_5 = np.array([3, 7, 11, 12, 15, 18, 20, 21, 27, 33])
asr_eight_6 = np.array([6, 8, 10, 16, 19, 23, 27, 29, 34, 38])
asr_eight_7 = np.array([3, 6, 9, 13, 18, 28, 32, 37, 41, 46])
asr_eight_8 = np.array([1, 9, 14, 24, 32, 37, 41, 43, 50, 55])
asr_eight_9 = np.array([5, 9, 17, 22, 29, 37, 43, 48, 54, 56])

# Plot ASR (8 bit mismatch) vs queries
plt.plot(queries, asr_eight_1, label='Stepsize: 0.1')
plt.plot(queries, asr_eight_2, label='Stepsize: 0.2')
plt.plot(queries, asr_eight_3, label='Stepsize: 0.3')
plt.plot(queries, asr_eight_4, label='Stepsize: 0.4')
plt.plot(queries, asr_eight_5, label='Stepsize: 0.5')
plt.plot(queries, asr_eight_6, label='Stepsize: 0.6')
plt.plot(queries, asr_eight_7, label='Stepsize: 0.7')
plt.plot(queries, asr_eight_8, label='Stepsize: 0.8')
plt.plot(queries, asr_eight_9, label='Stepsize: 0.9')
plt.title('ASR vs Queries (8  bit mismatch)')
plt.xlabel('Queries')
plt.ylabel('ASR (%)')
plt.legend()
plt.grid()
plt.show()

# ASR at 16 mismatched bit threshold
asr_sixteen_1 = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
asr_sixteen_2 = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
asr_sixteen_3 = np.array([0, 0, 0, 0, 0, 0, 1, 2, 2, 2])
asr_sixteen_4 = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 3])
asr_sixteen_5 = np.array([])
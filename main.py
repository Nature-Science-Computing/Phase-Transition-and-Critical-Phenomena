from library import swendsen_wang, plot
import matplotlib.pyplot as plt
import numpy as np
import pickle

L, beta = 8, 0.4

'''
############## Plots ##############
s_configs, g_configs = swendsen_wang(
    L, beta, num_sweeps=2 * 3 - 1, sc=False)

fig, axes = plt.subplots(2, 3)
for i, ax in enumerate(axes.flat):
    plot(ax, s_configs[i], g_configs[i])

fig.suptitle('Regular Swendsen Wang')
plt.show()

############## Plots Single Cluster ##############
s_configs, g_configs, s = swendsen_wang(
    L, beta, num_sweeps=2 * 3 - 1, sc=True)
s.append(None)

fig, axes = plt.subplots(2, 3)
for i, ax in enumerate(axes.flat):
    plot(ax, s_configs[i], g_configs[i], s=s[i])

fig.suptitle('Single Cluster Swendsen Wang')
plt.show()
'''


'''
############## Lattice Properties Calculation ##############
# Get lattice properties
avgs = swendsen_wang(L, beta, num_sweeps=10000, sc=True,
                     avg=True, print_progress=True)
print('---------------- Results ----------------')
print(
    'Energy E:\t\t({0:.3f} +- {1:.3f})\t(Expect (8x8): −1.219)'.format(avgs[0][0], avgs[0][1][-1]))
print('Magnetisation M:\t({0:.3f} +- {1:.3f})\t(Expect: 0)'.format(avgs[1][0], avgs[1][1][-1]))
print('-----------------------------------------')

# pickle.dump(avgs, open('results', 'wb'))
'''


############## Lattice Properties Results ##############
# avgs = pickle.load(open('results_8x8_b0.4_1500000_binning', 'rb'))
avgs = pickle.load(open('results_new', 'rb'))


fig, ax = plt.subplots()
for avg, label in zip(avgs, ['Energy', 'Magnetisation']):

    ax.plot(range(1, len(avg[1]) + 1), avg[1], label=label, linewidth=0.75)

ax.legend()
ax.grid()
ax.set(xlabel=fr'Bin Size $k$', ylabel=fr'Mean Bin Standard Deviation $<\sigma>$',
       title=fr'Error convergence')
'''
print('---------------- Results (10000 Runs) ----------------')
print('\t\t 1. Naive \t\t')
print(
    'Energy E:\t\t({0:.5f} +- {1:.5f})\t(Expect (8x8): −1.219)'.format(*avgs[0]))
print('Magnetisation M:\t({0:.5f} +- {1:.5f})\t(Expect: 0)'.format(*avgs[1]))
print('')
print('\t\t 2. Last Value from Binning \t\t')
print('Energy E:\t\t({0:.2f} +- {1:.2f})\t\t(Expect (8x8): −1.219)'.format(
    avgs[0][0], avgs[0][2][-1]))
print(
    'Magnetisation M:\t({0:.2f} +- {1:.2f})\t\t(Expect: 0)'.format(avgs[1][0], avgs[1][2][-1]))
print('-----------------------------------------')
'''
plt.show()


'''
############## Time Analysis: 500x500 Lattice (1 run) ##############

# Regular
Bond configuration:
--- 43.5486080647 seconds - --
Breadth First Search:
--- 26.1835689545 seconds - --
Entire Program
--- 72.5378448963 seconds - --

# Single Cluster
Bond configuration:
--- 43.3006548882 seconds - --
Breadth First Search:
--- 0.0002450943 seconds - -- 
Entire Program
--- 43.3037400246 seconds - --
'''

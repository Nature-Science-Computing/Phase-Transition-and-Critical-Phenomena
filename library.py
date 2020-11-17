import time
import numpy as np
from numpy.random import randint, choice
from matplotlib.patches import Circle

global is_timed
is_timed = False


def timeit(func):

    if is_timed or func.__name__ == 'swendsen_wang':
        def wrapper(*args, **kwargs):

            start_time = time.time()
            result = func(*args, **kwargs)
            end_time = time.time()

            print('Execution Time ({0}): {1:.5f} seconds'.format(
                func.__name__, end_time - start_time))

            return result

    else:
        def wrapper(*args, **kwargs):
            result = func(*args, **kwargs)
            return result

    return wrapper


@timeit
def swendsen_wang(L, beta, num_sweeps=1, flip_p=1 / 2, sc=False,
                  avg=False, print_progress=False):

    p = p_from_beta(beta)
    # Generate random spin configuration
    s_config = choice([True, False], size=(L, L))
    s_configs = [s_config]
    g_configs = []
    s = [] if sc else [None]
    avgs = [get_averages(s_config.flatten(), L)] if avg else None
    last_percentage = -1

    for i in range(num_sweeps):
        # Fixed lattice point for single cluster version
        if sc:
            s.append(randint(0, L**2))

        current_s_config = s_configs[-1].flatten()
        # Get bond configuration for the spin configuration
        if avg:
            g_configs = [g_config(current_s_config, p, L)]

        else:
            g_configs.append(g_config(current_s_config, p, L))
        # Get clusters from bond configuration
        clusters = breadth_first_search(current_s_config, g_configs[-1], s[-1])
        # Flip all clusters
        new_s_config = current_s_config.copy()
        for cluster in clusters:
            # Flip with 50% chance or 100% for single cluster version
            if choose(1 / 2) or s[-1] is not None:
                new_s_config[cluster] = np.invert(new_s_config[cluster])

        if avg:
            s_configs = [new_s_config.reshape(L, L)]
            avgs.append(get_averages(new_s_config, L))

        else:
            s_configs.append(new_s_config.reshape(L, L))

        if print_progress:
            percentage = int(np.floor(100 * i / num_sweeps))
            if percentage > last_percentage:
                print('{0: 3} %'.format(percentage))

            last_percentage = percentage

    if avg:

        results = [binning(observable) for observable in np.transpose(avgs)]

        return results

    else:
        g_configs.append(g_config(s_configs[-1].flatten(), p, L))
        if sc:
            return s_configs, g_configs, s

        else:
            return s_configs, g_configs


@timeit
def breadth_first_search(s_config, g, s=None):

    L = int(np.sqrt(len(s_config)))
    in_cluster = np.zeros(L**2, dtype=np.bool)
    clusters = []

    def find_cluster(i):
        # If already in a cluster continue with next lattice point
        if in_cluster[i]:
            return None

        # If not in cluster make a new cluster(-list) and add it to it
        in_cluster[i] = True
        cluster = [i]
        # Go through all points in the cluster
        for c in cluster:
            # Check all neighbours and add them to the list
            # (and extent the loop)
            for mu in [-2, -1, 1, 2]:
                j = neighbour(c, mu, L)
                # If the neighbour is not connected by a bond or already
                # in the list go to next neighbour
                if g.get_vertex(c, j) != 1 or j in cluster:
                    continue

                # If not add it to cluster lists
                in_cluster[j] = True
                cluster.append(j)

        return cluster

    if s is not None:
        cluster = find_cluster(s)
        clusters.append(cluster)

    else:
        # Go through all lattice points
        for i in range(len(s_config)):
            cluster = find_cluster(i)
            if cluster is not None:
                clusters.append(cluster)

    return clusters


@timeit
def get_averages(s_config, L):

    M = np.sum(s_config) - np.sum(np.invert(s_config))
    E = 0
    for i, s_i in enumerate(s_config):
        for mu in [1, 2]:
            j = neighbour(i, mu, L)
            s_i_s_j = s_i ^ s_config[j]
            E = E + 1 if s_i_s_j else E - 1

    avg = np.array([E, M]) / L**2

    return avg


def binning(time_series, error=0.02, conv_criteria=3):

    convergent_block_sizes = 0
    vars_ = []
    mean = np.mean(time_series)

    # The bin size should not go below about 100
    for k in range(1, len(time_series)):
        blocks = split_into_blocks(time_series, k)
        vars_.append(np.var([np.mean(block)
                             for block in blocks]) / len(blocks[0]))

        if len(vars_) == 1 or vars_[-2] == 0:
            continue

        elif abs(vars_[-1] / vars_[-2] - 1) <= error:
            convergent_block_sizes += 1

            if convergent_block_sizes >= conv_criteria:
                return mean, np.sqrt(vars_)

        else:
            convergent_block_sizes = 0

    print('No convergence for the error could been reached.')
    return mean, np.std(time_series) / np.sqrt(len(time_series)), np.sqrt(vars_)


def split_into_blocks(time_series, k):

    divisblity_index = len(time_series) % k

    if divisblity_index == 0:
        blocks = np.split(time_series, k)

    else:
        blocks = np.split(time_series[:-divisblity_index], k)

    return blocks


def neighbour(i, mu, L):

    row, col = np.unravel_index(i, (L, L))

    # Go Left
    if mu == 1:
        col = col + 1 if col < L - 1 else 0

    # Go Right
    elif mu == -1:
        col = col - 1 if col > 0 else L - 1

    # Go Up
    elif mu == 2:
        row = row - 1 if row > 0 else L - 1

    # Go Down
    elif mu == -2:
        row = row + 1 if row < L - 1 else 0

    return np.ravel_multi_index((row, col), (L, L))


def choose(p):

    return choice([1, 0], p=[p, 1 - p])


def p_from_beta(beta):

    if beta == 0:
        p = 0

    else:
        p = 1 - np.exp(-2 * beta)

    return p


class g_config():

    @timeit
    def __init__(self, s_config, p, L):

        self.p = p
        self.L = L
        self.g = [-1 * np.ones(self.L ** 2, dtype=np.short), -
                  1 * np.ones(self.L ** 2, dtype=np.short)]

        self.set_bonds(s_config, p)

    def set_bonds(self, s_config, p):

        for i, s_i in enumerate(s_config):
            for mu in [1, 2]:
                j = neighbour(i, mu, self.L)
                if self.get_vertex(i, j) == -1:
                    value = 0 if s_i != s_config[j] else choose(p)
                    self.set_vertex(i, j, value)

    def index_transform(self, i, j):

        i, j = np.sort([i, j])
        if np.floor(i / self.L) == np.floor(j / self.L):
            list_idx = 0
            idx = j if j - i == 1 else i

        else:
            list_idx = 1
            idx = j if j - i == self.L else i

        return int(list_idx), int(idx)

    def set_vertex(self, i, j, value):

        l, idx = self.index_transform(i, j)

        self.g[l][idx] = value

    def get_vertex(self, i, j):

        l, idx = self.index_transform(i, j)

        return self.g[l][idx]

    def __str__(self):

        string = str(self.g[0]) + '\n' + str(self.g[1])

        return string


def plot(ax, s_config, g_config, s=None):

    L = len(s_config)
    # Plot spin configuration
    ax.imshow(s_config)

    # Plot a red dot for the single cluster lattice point
    if s is not None:
        x, y = np.flip(np.unravel_index(s, (L, L)))
        dot = Circle((x, y), radius=0.25, color='red')
        ax.add_patch(dot)

    # Plot all cluster bonds
    for i, _ in enumerate(s_config.flatten()):
        for mu in [-2, 1, 1, 2]:
            j = neighbour(i, mu, L)
            i, j = np.sort([i, j])
            if g_config.get_vertex(i, j):
                x_i, y_i = np.flip(np.unravel_index(i, (L, L)))
                x_j, y_j = np.flip(np.unravel_index(j, (L, L)))

                if j - i == 1 or j - i == L:
                    ax.plot([x_i, x_j], [y_i, y_j], color='k', linewidth=2)

                elif j - i > 1 and j - i < L:
                    ax.plot([x_i, -0.5], [y_i, y_i], color='k', linewidth=2)
                    ax.plot([x_j, x_j + 0.5], [y_j, y_j],
                            color='k', linewidth=2)

                else:
                    ax.plot([x_i, x_i], [y_i, -0.5], color='k', linewidth=2)
                    ax.plot([x_j, x_j], [y_j, y_j + 0.5],
                            color='k', linewidth=2)

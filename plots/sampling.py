import random
import numpy as np
import matplotlib.pyplot as plt

from memory.tree import SumTree


def sample_segments(tree, batch_size):
    segment = tree.total / batch_size

    priorities = []
    for i in range(batch_size):
        a, b = segment * i, segment * (i + 1)

        cumsum = random.uniform(a, b)
        data_idx, priority, sample_idx = tree.get(cumsum)

        priorities.append(priority)

    return priorities


def sample_cumsum(tree, batch_size):
    cumsums = np.random.uniform(0, tree.total, size=batch_size)

    priorities = []
    for i, cumsum in enumerate(cumsums):
        data_idx, priority, sample_idx = tree.get(cumsum)
        priorities.append(priority)

    return priorities


def main():
    tree = SumTree(size=1000)

    for i in range(1000):
        p = random.uniform(0, 50)
        tree.add(p, p)

    segments = sum([sample_segments(tree, 64) for _ in range(5000)], [])
    cumsums = sum([sample_cumsum(tree, 64) for _ in range(5000)], [])

    plt.figure(figsize=(12, 8))
    plt.hist(cumsums, bins=64, label="sample from [0, total]")
    plt.hist(segments, bins=64, label="sample from segments", alpha=0.8)
    plt.xlabel("Priority")
    plt.ylabel("Frequency")
    plt.legend()
    plt.savefig("sampling_approaches.jpg", dpi=200, bbox_inches='tight')


if __name__ == "__main__":
    main()
import numpy as np
from math import ceil
import torch


def convert(partition):
    """
    Convert a binary matrix representation of a partition to a list of parts.

    Parameters:
    - matrix_partition (np.ndarray): Binary matrix representation of a partition.

    Returns:
    - list: List of parts, where each part is a list of integers.
    """
    result = [[] for _ in range(partition.shape[0])]
    ones = np.argwhere(partition == 1)
    for index in ones:
        result[index[0]].append(index[1]+1)
    return result


def convert_to_seq(partition):
    """
    Convert a binary matrix representation of a partition to a sequence.

    Parameters:
    - matrix_partition (np.ndarray): Binary matrix representation of a partition.

    Returns:
    - list: Sequence of integers.
    """
    seq = []
    for i in range(partition.shape[1]):
        flag = True
        for j in range(partition.shape[0]):
            if partition[j][i] == 1:
                seq.append(j)
                flag = False
                break
        if flag:
            break
    return seq


def is_schur(partition):
    """
    Check if a partition (in binary matrix form) satisfies the Schur property.

    Parameters:
    - matrix_partition (np.ndarray): Binary matrix representation of a partition.

    Returns:
    - bool: True if the partition satisfies the Schur property, False otherwise.
    """
    #partition = convert(partition)
    if all([len(x) == 0 for x in partition]):
        return False
    for part in partition:
        for i, a in enumerate(part):
            if has_two_sum(part, a):
                return False
    return True


def convert_seq_to_parts(seq, n):
    """
    Convert a sequence to a list of parts.

    Parameters:
    - seq (list): Sequence of integers.
    - num_partitions (int): Number of partitions.

    Returns:
    - list: List of parts, where each part is a list of integers.
    """
    parts = [[] for _ in range(n)]
    for j, i in enumerate(seq):
        if i < n:
            parts[int(i)].append(int(j)+1)
    return parts


def seq_is_schur(seq, n):
    """
    Check if a sequence satisfies the Schur property.

    Parameters:
    - seq (list): Sequence of integers.
    - num_partitions (int): Number of partitions.

    Returns:
    - bool: True if the sequence satisfies the Schur property, False otherwise.
    """
    partition = convert_seq_to_parts(seq, n)
    return is_schur(partition)


def has_two_sum(nums: list, target: int):
    """
    Check if there exist two elements in a list that sum up to a target value.

    Parameters:
    - nums (list): List of integers.
    - target (int): Target sum value.

    Returns:
    - bool: True if there exist two elements summing up to the target, False otherwise.
    """
    if len(nums) < 2:
        return []
    start = 0
    end = len(nums) - 1

    while end >= start:
        if nums[start] + nums[end] > target:
            end -= 1
        elif nums[start] + nums[end] < target:
            start += 1
        else:
            return True
    return False


def generate_incidence(n):
    hyperedges = []
    for i in range(n, 1, -1):
        for j in range(1, ceil(i/2)+1):
            if 2*j == i:
                hyper = (j, i)
            else:
                hyper = (min(j, i-j), max(j, i-j), i)
            if hyper not in hyperedges:
                hyperedges.append(tuple(hyper))

    nodes = list(range(1, n+1))
    incidence_matrix = np.zeros((len(nodes), len(hyperedges)))
    for i, edge in enumerate(hyperedges):
        for j in edge:
            incidence_matrix[j-1, i] = 1

    return torch.tensor(incidence_matrix)

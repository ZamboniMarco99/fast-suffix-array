# Implementation based on https://www.cs.helsinki.fi/u/tpkarkka/publications/jacm05-revised.pdf and https://mailund.dk/posts/skew-python-go/

import numpy as np
import numba
from typing import Tuple


@numba.jit()
def merge(x: np.array, SA12: np.array, SA3: np.array) -> np.array:
    "Merge the suffixes in sorted SA12 and SA3."
    ISA = np.zeros((len(x),), dtype='int')
    for i in range(len(SA12)):
        ISA[SA12[i]] = i
    SA = np.zeros((len(x),), dtype='int')
    idx = 0
    i, j = 0, 0
    while i < len(SA12) and j < len(SA3):
        if less(x, SA12[i], SA3[j], ISA):
            SA[idx] = SA12[i]
            idx += 1
            i += 1
        else:
            SA[idx] = SA3[j]
            idx += 1
            j += 1
    if i < len(SA12):
        SA[idx:len(SA)] = SA12[i:]
    elif j < len(SA3):
        SA[idx:len(SA)] = SA3[j:]
    return SA


@numba.jit()
def u_idx(i: int, m: int) -> int:
    "Map indices in u back to indices in the original string."
    if i < m:
        return 1 + 3 * i
    else:
        return 2 + 3 * (i - m - 1)


@numba.jit()
def safe_idx(x: np.array, i: int) -> int:
    "Hack to get zero if we index beyond the end."
    return 0 if i >= len(x) else x[i]


@numba.jit()
def symbcount(x: np.array, asize: int) -> np.array:
    "Count how often we see each character in the alphabet."
    counts = np.zeros((asize,), dtype="int")
    for c in x:
        counts[c] += 1
    return counts


@numba.jit()
def cumsum(counts: np.array) -> np.array:
    "Compute the cumulative sum from the character count."
    res = np.zeros((len(counts, )), dtype='int')
    acc = 0
    for i, k in enumerate(counts):
        res[i] = acc
        acc += k
    return res


@numba.jit()
def bucket_sort(x: np.array, asize: int,
                idx: np.array, offset: int = 0) -> np.array:
    "Sort indices in idx according to x[i + offset]."
    sort_symbs = np.array([safe_idx(x, i + offset) for i in idx])
    counts = symbcount(sort_symbs, asize)
    buckets = cumsum(counts)
    out = np.zeros((len(idx),), dtype='int')
    for i in idx:
        bucket = safe_idx(x, i + offset)
        out[buckets[bucket]] = i
        buckets[bucket] += 1
    return out


@numba.jit()
def radix3(x: np.array, asize: int, idx: np.array) -> np.array:
    "Sort indices in idx according to their first three letters in x."
    idx = bucket_sort(x, asize, idx, 2)
    idx = bucket_sort(x, asize, idx, 1)
    return bucket_sort(x, asize, idx)


@numba.jit()
def triplet(x: np.array, i: int) -> Tuple[int, int, int]:
    "Extract the triplet (x[i],x[i+1],x[i+2])."
    return safe_idx(x, i), safe_idx(x, i + 1), safe_idx(x, i + 2)


@numba.jit()
def collect_alphabet(x: np.array, idx: np.array) -> Tuple[np.array, int]:
    "Map the triplets starting at idx to a new alphabet."
    alpha = np.zeros((len(x),), dtype='int')
    value = 1
    last_trip = -1, -1, -1
    for i in idx:
        trip = triplet(x, i)
        if trip != last_trip:
            value += 1
            last_trip = trip
        alpha[i] = value
    return alpha, value - 1


@numba.jit()
def build_u(x: np.array, alpha: np.array) -> np.array:
    "Construct u string, using 1 as central sentinel."
    a = np.array([alpha[i] for i in range(1, len(x), 3)] +
                 [1] +
                 [alpha[i] for i in range(2, len(x), 3)])
    return a


@numba.jit()
def less(x: np.array, i: int, j: int, ISA: np.array) -> bool:
    "Check if x[i:] < x[j:] using the inverse suffix array for SA12."
    a: int = safe_idx(x, i)
    b: int = safe_idx(x, j)
    if a < b: return True
    if a > b: return False
    if i % 3 != 0 and j % 3 != 0: return ISA[i] < ISA[j]
    return less(x, i + 1, j + 1, ISA)


@numba.jit()
def skew_rec(x: np.array, asize: int) -> np.array:
    "skew/DC3 SA construction algorithm."

    SA12 = np.array([i for i in range(len(x)) if i % 3 != 0])

    SA12 = radix3(x, asize, SA12)
    new_alpha, new_asize = collect_alphabet(x, SA12)
    if new_asize < len(SA12):
        # Recursively sort SA12
        u = build_u(x, new_alpha)
        sa_u = skew_rec(u, new_asize + 2)
        m = len(sa_u) // 2
        SA12 = np.array([u_idx(i, m) for i in sa_u if i != m])

    if len(x) % 3 == 1:
        SA3 = np.array([len(x) - 1] + [i - 1 for i in SA12 if i % 3 == 1])
    else:
        SA3 = np.array([i - 1 for i in SA12 if i % 3 == 1])
    SA3 = bucket_sort(x, asize, SA3)
    return merge(x, SA12, SA3)


def skew_dna(x: str) -> np.array:
    str_to_int = {
        "$": 0,  # End of string
        "A": 1,
        "C": 2,
        "G": 3,
        "N": 4,  # Unknown nucleotide
        "T": 5,
    }
    return skew_rec(np.array([str_to_int[y] for y in x]), 6)

if __name__ == "__main__":
    dna_string = input()
    suffix_array = skew_dna(dna_string)
    print(suffix_array)

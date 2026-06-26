def FBED(target, n_vars, n_states, max_max_cond_size):
    mb = np.full(n_vars, -1, dtype=int)
    for i in range(n_vars):
        if n_states[i] > 1 and n_states[target] > 1:
            mb[i] = 0

    mb[target] = -1

    n_conds = 0
    cond = np.full(max_max_cond_size, -1, dtype=int)
    R = np.zeros(n_vars, dtype=int)
    dep = np.zeros(n_vars, dtype=float)

    K = 1
    K_cur = 0

    # Growth Phase
    while K_cur <= K:
        stop1 = 1

        if n_conds < max_max_cond_size:
            n_Rs = 0
            for i in range(n_vars):
                if mb[i] == 0:
                    R[i] = 1
                    n_Rs += 1

            tag = 1
            while n_Rs > 0 and tag == 1:
                tag = 0
                if n_conds < max_max_cond_size:
                    dep.fill(0.0)
                    for i in range(n_vars):
                        if R[i] == 1:
                            dep[i] = compute_dep(i, target, cond)
                            if dep[i] >= 1.0:
                                tag = 1
                                stop1 = 0

                    if tag == 1:
                        aux = k_greedy(dep)
                        mb[aux] = 1
                        cond[n_conds] = aux
                        n_conds += 1
                        for i in range(n_vars):
                            if i == aux or dep[i] <= -1.0:
                                if R[i] == 1:
                                    R[i] = 0
                                    n_Rs -= 1
        K_cur += 1
        if stop1:
            break

    dep.fill(0.0)
    for i in range(n_vars):
        if mb[i] == 1:
            for j in range(n_conds):
                if cond[j] == i:
                    cond[j:n_conds - 1] = cond[j + 1:n_conds]
                    cond[n_conds - 1] = -1
                    break
            dep[i] = compute_dep(i, target, cond)
            if dep[i] <= -1.0:
                n_conds -= 1
                mb[i] = 0
            else:
                cond[n_conds - 1] = i

    mb[mb == -1] = 0
    report_mb(target, mb)
    return mb

import numpy as np
from itertools import combinations

def compute_dep(var, target, cond):
    # TODO: Replace this with your own dependency measure
    return np.random.uniform(-2.0, 2.0)

def min_dep(var, target, other, pc, sep2):
    # Simulated minimum dependency measure
    return compute_dep(var, target, [])

def next_cond_index(n, k, idx):
    # Generate next combination index in lexicographic order
    for i in reversed(range(k)):
        if idx[i] != i + n - k:
            break
    else:
        return 1  # Done
    idx[i] += 1
    for j in range(i+1, k):
        idx[j] = idx[j-1] + 1
    return 0

def check_spouse_removal(s_y, last_added, Z, max_cond_size, k_cond, sep, sep_tag):
    n_codes = len(Z)
    for cond_size in range(min(k_cond, n_codes)+1):
        if cond_size + 1 > max_cond_size:
            continue
        idx = list(range(cond_size))
        while True:
            cond = [-1] * max_cond_size
            cond[0] = last_added
            for i in range(cond_size):
                cond[i+1] = Z[idx[i]]
            if compute_dep(s_y, last_added, [c for c in cond if c != -1]) <= -1.0:
                if not sep_tag[s_y]:
                    sep[s_y][:] = cond[:]
                    sep_tag[s_y] = 1
                return True
            if next_cond_index(n_codes, cond_size, idx):
                break
    return False

def report_mb(target, mb):
    print(f"Markov Blanket for target {target}: {[i for i, v in enumerate(mb) if v == 1]}")

# def BAMB(target, n_vars, n_states, max_max_cond_size, k_condition):
#     pc = np.full(n_vars, 0)
#     mb = np.zeros(n_vars, dtype=int)
#     dep = np.zeros(n_vars, dtype=float)
#     sep = [[-1]*max_max_cond_size for _ in range(n_vars)]
#     sep2 = [-1]*max_max_cond_size
#     cond = [-1]*max_max_cond_size
#     sep_tag = [0]*n_vars
#
#     sp_dep_rank = [[] for _ in range(n_vars)]
#     spouse = [[] for _ in range(n_vars)]
#     pc_dep_rank = []
#
#     for i in range(n_vars):
#         if n_states[i] <= 1 or n_states[target] <= 1:
#             pc[i] = -1
#
#     rank = []
#     for i in range(n_vars):
#         if pc[i] == 0:
#             dep[i] = min_dep(i, target, -1, pc, sep2)
#             if dep[i] <= -1.0:
#                 pc[i] = -1
#                 sep_tag[i] = 1
#                 sep[i] = sep2[:]
#             elif dep[i] >= 1.0:
#                 rank.append((i, dep[i]))
#
#     rank = [i for i, _ in sorted(rank, key=lambda x: -x[1])]
#
#     for last_added in rank:
#         pc[last_added] = 1
#         pc[last_added] = 0
#         if min_dep(last_added, target, -1, pc, sep2) <= -1.0:
#             sep[last_added] = sep2[:]
#             pc[last_added] = -1
#             continue
#         else:
#             pc[last_added] = 1
#             pc_dep_rank.append(last_added)
#
#         for j in range(n_vars):
#             if pc[j] == 1 and j != last_added:
#                 pc[j] = 0
#                 dep[j] = min_dep(j, target, last_added, pc, sep2)
#                 pc[j] = 1
#                 if dep[j] <= -1.0:
#                     sep[j] = sep2[:]
#                     pc[j] = -1
#                     if j in pc_dep_rank:
#                         pc_dep_rank.remove(j)
#
#         dep.fill(0.0)
#         for j in range(n_vars):
#             if j != target and pc[j] != 1:
#                 cond = sep[j][:]
#                 n_conds = sum(1 for x in cond if x != -1)
#                 if last_added not in cond and n_conds < max_max_cond_size:
#                     cond[n_conds] = last_added
#                     dep[j] = compute_dep(j, target, cond)
#                     if dep[j] <= -1.0 and not sep_tag[j]:
#                         sep[j] = cond[:]
#                         sep_tag[j] = 1
#
#         pre_sps = sorted([(i, d) for i, d in enumerate(dep) if d > 0], key=lambda x: -x[1])
#         sp_dep_rank[last_added] = [i for i, _ in pre_sps]
#
#         while sp_dep_rank[last_added]:
#             y = sp_dep_rank[last_added].pop(0)
#             spouse[last_added].append(y)
#
#             for k in reversed(range(len(spouse[last_added]))):
#                 s_y = spouse[last_added][k]
#                 Z = [sp for idx, sp in enumerate(spouse[last_added]) if idx != k]
#                 if check_spouse_removal(s_y, last_added, Z, max_max_cond_size, k_condition, sep, sep_tag):
#                     spouse[last_added].pop(k)
#
#     for i in range(n_vars):
#         mb[i] = 0
#     for i in pc_dep_rank:
#         mb[i] = 1
#         for s in spouse[i]:
#             mb[s] = 1
#
#     report_mb(target, mb)
#     return mb

import numpy as np
from itertools import combinations


def min_dep(var, target, other, pc, sep2):
    # Simulated minimum dependency measure
    return compute_dep(var, target, [])

def next_cond_index(n, k, idx):
    # Generate next combination index in lexicographic order
    for i in reversed(range(k)):
        if idx[i] != i + n - k:
            break
    else:
        return 1  # Done
    idx[i] += 1
    for j in range(i+1, k):
        idx[j] = idx[j-1] + 1
    return 0

def check_spouse_removal(s_y, last_added, Z, max_cond_size, k_cond, sep, sep_tag):
    n_codes = len(Z)
    for cond_size in range(min(k_cond, n_codes)+1):
        if cond_size + 1 > max_cond_size:
            continue
        idx = list(range(cond_size))
        while True:
            cond = [-1] * max_cond_size
            cond[0] = last_added
            for i in range(cond_size):
                cond[i+1] = Z[idx[i]]
            if compute_dep(s_y, last_added, [c for c in cond if c != -1]) <= -1.0:
                if not sep_tag[s_y]:
                    sep[s_y][:] = cond[:]
                    sep_tag[s_y] = 1
                return True
            if next_cond_index(n_codes, cond_size, idx):
                break
    return False

def report_mb(target, mb):
    print(f"Markov Blanket for target {target}: {[i for i, v in enumerate(mb) if v == 1]}")

def BAMB(target, n_vars, n_states, max_max_cond_size, k_condition):
    pc = np.full(n_vars, 0)
    mb = np.zeros(n_vars, dtype=int)
    dep = np.zeros(n_vars, dtype=float)
    sep = [[-1]*max_max_cond_size for _ in range(n_vars)]
    sep2 = [-1]*max_max_cond_size
    cond = [-1]*max_max_cond_size
    sep_tag = [0]*n_vars

    sp_dep_rank = [[] for _ in range(n_vars)]
    spouse = [[] for _ in range(n_vars)]
    pc_dep_rank = []

    for i in range(n_vars):
        if n_states[i] <= 1 or n_states[target] <= 1:
            pc[i] = -1

    rank = []
    for i in range(n_vars):
        if pc[i] == 0:
            dep[i] = min_dep(i, target, -1, pc, sep2)
            if dep[i] <= -1.0:
                pc[i] = -1
                sep_tag[i] = 1
                sep[i] = sep2[:]
            elif dep[i] >= 1.0:
                rank.append((i, dep[i]))

    rank = [i for i, _ in sorted(rank, key=lambda x: -x[1])]

    for last_added in rank:
        pc[last_added] = 1
        pc[last_added] = 0
        if min_dep(last_added, target, -1, pc, sep2) <= -1.0:
            sep[last_added] = sep2[:]
            pc[last_added] = -1
            continue
        else:
            pc[last_added] = 1
            pc_dep_rank.append(last_added)

        for j in range(n_vars):
            if pc[j] == 1 and j != last_added:
                pc[j] = 0
                dep[j] = min_dep(j, target, last_added, pc, sep2)
                pc[j] = 1
                if dep[j] <= -1.0:
                    sep[j] = sep2[:]
                    pc[j] = -1
                    if j in pc_dep_rank:
                        pc_dep_rank.remove(j)

        dep.fill(0.0)
        for j in range(n_vars):
            if j != target and pc[j] != 1:
                cond = sep[j][:]
                n_conds = sum(1 for x in cond if x != -1)
                if last_added not in cond and n_conds < max_max_cond_size:
                    cond[n_conds] = last_added
                    dep[j] = compute_dep(j, target, cond)
                    if dep[j] <= -1.0 and not sep_tag[j]:
                        sep[j] = cond[:]
                        sep_tag[j] = 1

        pre_sps = sorted([(i, d) for i, d in enumerate(dep) if d > 0], key=lambda x: -x[1])
        sp_dep_rank[last_added] = [i for i, _ in pre_sps]

        while sp_dep_rank[last_added]:
            y = sp_dep_rank[last_added].pop(0)
            spouse[last_added].append(y)

            for k in reversed(range(len(spouse[last_added]))):
                s_y = spouse[last_added][k]
                Z = [sp for idx, sp in enumerate(spouse[last_added]) if idx != k]
                if check_spouse_removal(s_y, last_added, Z, max_max_cond_size, k_condition, sep, sep_tag):
                    spouse[last_added].pop(k)

    for i in range(n_vars):
        mb[i] = 0
    for i in pc_dep_rank:
        mb[i] = 1
        for s in spouse[i]:
            mb[s] = 1

    report_mb(target, mb)
    return mb


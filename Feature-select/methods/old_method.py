# def compute_dep_old(var, target, cond, data, n_states, n_cases):
#     n_cond_states = 1
#     cond = [c for c in cond if c != -1]
#     for c in cond:
#         n_cond_states *= n_states[c]
#     # if n_cond_states * (n_states[target] - 1) * (n_states[var] - 1) * N_CASES_PER_DF > n_cases:
#     #     return 0.0
#
#     ss_cond = np.zeros(n_cond_states, dtype=int)
#     ss_target = np.zeros((n_cond_states, n_states[target]), dtype=int)
#     ss_var = np.zeros((n_cond_states, n_states[var]), dtype=int)
#     ss = np.zeros((n_cond_states, n_states[target], n_states[var]), dtype=int)
#
#     for i in range(n_cases):
#         cond_state = 0
#         for c in cond:
#             cond_state = cond_state * n_states[c] + data[i, c]
#         ss_cond[cond_state] += 1
#         ss_target[cond_state][data[i, target]] += 1
#         ss_var[cond_state][data[i, var]] += 1
#         ss[cond_state][data[i, target]][data[i, var]] += 1
#
#     statistic = 0.0
#     for i in range(n_cond_states):
#         if ss_cond[i] > 0:
#             for j in range(n_states[target]):
#                 if ss_target[i][j] > 0:
#                     for k in range(n_states[var]):
#                         if ss[i][j][k] > 0:
#                             expected = (ss_target[i][j] * ss_var[i][k]) / ss_cond[i]
#                             statistic += ss[i][j][k] * (log(ss[i][j][k]) - log(expected))
#     statistic *= 2.0
#
#     df = 0
#     for i in range(n_cond_states):
#         if ss_cond[i] > 0:
#             df_target = np.sum(ss_target[i] > 0)
#             df_var = np.sum(ss_var[i] > 0)
#             df += (df_target - 1) * (df_var - 1)
#     if df <= 0:
#         df = 1
#
#     dep = gammaincc(0.5 * df, 0.5 * statistic)
#     if dep <= alpha:
#         dep = 2.0 - dep
#         if dep == 2.0:
#             dep += 2.0 +statistic / df
#         return dep
#     else:
#         dep = -1.0 - dep
#         if dep == -2.0:
#             dep -= -2.0 -statistic / df
#         return dep

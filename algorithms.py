def baseline_uncons(obs_util, grps, k, m, pr=False):
    n = np.sum([len(g) for g in grps])
    all = [i for i in range(n)]

    sol = [] #()

    for i in range(k):
        utils = marg_F_mult(sol, all, obs_util, m)
        tmp_sol = get_top_n(all, utils, 1)

        assert(len(tmp_sol) == 1)
        ind = list(tmp_sol)[0]
        all.remove(ind)

        sol.append(ind)

    assert(len(sol)==k)
    return sol

def algo_disj(obs_util, grps, k, m): # lat_util = None):
    p = len(grps)
    n = np.sum([len(g) for g in grps])
    grp_lists = [list(grps[t]) for t in range(p)]

    def count_sol_in_attr(sol, obs_util):
        k_pr = np.array([0.0 for j in range(m)])

        for j in range(m):
            for i in sol_tmp:
                k_pr[j] += (obs_util[i][j] > 0)

        k_pr *= float(float(k)/np.sum(k_pr))

        return k_pr

    #####################################

    obs_util_grp_a = copy.deepcopy(obs_util)
    for i in grps[1]: obs_util_grp_a[i] *= 0
    sol_tmp = baseline_greedy_uncons(obs_util_grp_a, grps, k * len(grps[0]) // n, m, pr=False)
    k_pr = count_sol_in_attr(sol_tmp, obs_util)

    ####################################

    sol = []

    for j in range(m):
        for t in range(p):
            k_int = int(k_pr[j] * len(grps[t]) / n)
            utils = [obs_util[i][j] for i in grps[t]]
            tmp_sol = get_top_n(grps[t], utils, k_int)
            for tmp in tmp_sol:
                sol.append(tmp)
    sol = set(sol)

    if len(sol) < k:
        unselec = []
        for i in range(n):
            if i not in sol:
                unselec.append(i)

        utils = marg_F_mult(sol, unselec, obs_util, m)
        tmp_sol = get_top_n(unselec, utils, k - len(sol))

        for i in tmp_sol:
            sol.add(i)
            if len(sol) == k:
                break
        sol = list(sol)

    assert(len(sol)==k)

    return sol

def algo_iid(obs_util, grps, k, m, emerging_artists=None, ratio=-1, pr = False, lat_util=None):
    p = len(grps)
    n = np.sum([len(g) for g in grps])
    grp_lists = [list(grps[t]) for t in range(p)]

    sol = []

    cnt = 0
    pr=False

    for ind in range(k):
        if ratio <= 0:
            t = int(rng.random() < len(grps[1]) / n)
        else:
            t = int(rng.random() < ratio)

        cnt += t

        if len(grp_lists[t]) == 0:
            t = 1-t

        if ind <= np.sqrt(k):
            t = 0
            utils = marg_linear_mult(sol, grp_lists[t], obs_util, m, pr=False)
            tmp_sol = get_top_n(grp_lists[t], utils, 1)
        elif ind <= np.sqrt(k)*len(grp_lists[1])/len(grp_lists[0]):
            t = 1
            utils = marg_linear_mult(sol, grp_lists[t], obs_util, m, pr=False)
            tmp_sol = get_top_n(grp_lists[t], utils, 1)
        else:
            tmp_grps = rng.choice(grp_lists[t], max(1, int(len(grp_lists[t])*0.8)), replace=False)
            utils = marg_F_mult(sol, tmp_grps, obs_util, m, pr=False)
            tmp_sol = get_top_n(tmp_grps, utils, 1)

        assert(len(tmp_sol) == 1)

        ind = list(tmp_sol)[0]
        assert(ind in grp_lists[t])

        grp_lists[t].remove(ind)

        sol.append(ind)

    assert(len(sol)==k)
    return sol

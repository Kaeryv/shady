import numpy as np

'''
    L-SHADE Matlab implementation translated in python.
    From https://sites.google.com/site/tanaberyoji/software
    By Nicolas Roy (@Kaeryv) 
    Initial hack: dec-23
'''

def bound_constraint(vi, pop, lu):
    NP, D = pop.shape

    xl = np.tile(lu[0, :], (NP, 1))
    pos = vi < xl
    vi[pos] = (pop[pos] + xl[pos]) / 2

    xu = np.tile(lu[1, :], (NP, 1))
    pos = vi > xu
    vi[pos] = (pop[pos] + xu[pos]) / 2

    return vi

def update_archive(archive, pop, funvalue):
    if archive['NP'] == 0:
        return

    if pop.shape[0] != funvalue.shape[0]:
        raise ValueError('Check input shapes')

    # Method 2: Remove duplicate elements
    pop_all = np.vstack((archive['pop'], pop))
    funvalues = np.hstack((archive['funvalues'], funvalue))
    _, unique_indices = np.unique(pop_all, axis=0, return_index=True)

    if len(unique_indices) < pop_all.shape[0]:
        pop_all = pop_all[unique_indices, :]
        funvalues = funvalues[unique_indices]

    if pop_all.shape[0] <= archive['NP']:
        # Add all new individuals
        archive['pop'] = pop_all
        archive['funvalues'] = funvalues
    else:
        # Randomly remove some solutions
        rndpos = np.random.permutation(pop_all.shape[0])[:archive['NP']]

        archive['pop'] = pop_all[rndpos, :]
        archive['funvalues'] = funvalues[rndpos]

    return archive

def gnR1R2(NP1, NP2, r0):
    NP0 = len(r0)

    r1 = np.random.randint(0, NP1, NP0)
    for i in range(1000):
        pos = (r1 == r0)
        if np.sum(pos) == 0:
            break
        else:
            r1[pos] = np.random.randint(0, NP1, np.sum(pos))

    else:
        raise ValueError('Cannot generate r1 in 1000 iterations')

    r2 = np.random.randint(0, NP2, NP0)
    for i in range(1000):
        pos = ((r2 == r1) | (r2 == r0))
        if np.sum(pos) == 0:
            break
        else:
            r2[pos] = np.random.randint(0, NP2, np.sum(pos))

    else:
        raise ValueError('Cannot generate r2 in 1000 iterations')

    return r1, r2

from copy import copy
def optimize_lshade(fitness_function, lu, max_nfes=80000):
    profile = []
    ND = lu.shape[1]

    p_best_rate = 0.11
    arc_rate = 1.4
    memory_size = 5
    pop_size = 4 * ND

    max_pop_size = copy(pop_size)
    min_pop_size = 4.0

    popold = lu[0, :] + np.random.rand(pop_size, ND) * (lu[1, :] - lu[0, :])

    pop = popold
    fitness = fitness_function(pop)
    nfes = 0

    # Monitoring
    bsf_location = np.argmin(fitness)
    bsf_fit_var = fitness[bsf_location]
    bsf_solution = pop[bsf_location].copy()

    memory_sf = np.full((memory_size), 0.5)
    memory_cr = np.full((memory_size), 0.5)
    memory_pos = 0

    archive = {
        'NP': int(arc_rate * pop_size), 
        'pop': np.zeros((0, ND)), 
        'funvalues': np.zeros((0))
    }

    while nfes < max_nfes:
        pop = popold
        sorted_index = np.argsort(fitness)

        mem_rand_index = np.floor(memory_size * np.random.rand(pop_size)).astype(int)
        mu_sf = memory_sf[mem_rand_index]
        mu_cr = memory_cr[mem_rand_index]

        cr = np.random.normal(mu_cr, 0.1)
        term_pos = np.where(mu_cr == -1)
        cr[term_pos] = 0
        cr = np.clip(cr, 0, 1)

        sf = mu_sf + 0.1 * np.tan(np.pi * (np.random.rand(pop_size) - 0.5))
        pos = np.where(sf <= 0)

        while len(pos[0]) > 0:
            sf[pos] = mu_sf[pos] + 0.1 * np.tan(np.pi * (np.random.rand(len(pos[0])) - 0.5))
            pos = np.where(sf <= 0)

        sf = np.minimum(sf, 1)

        r0 = np.arange(pop_size)
        pop_all = np.vstack((pop, archive['pop']))
        r1, r2 = gnR1R2(pop_size, pop_all.shape[0], r0)

        pNP = max(round(p_best_rate * pop_size), 2)
        randindex = np.floor(np.random.rand(pop_size) * pNP)
        randindex = np.maximum(0, randindex).astype(int)
        pbest = pop[sorted_index[randindex], :]

        vi = pop + sf[:, np.newaxis] * (pbest - pop + pop[r1, :] - pop_all[r2, :])
        vi = bound_constraint(vi, pop, lu)

        mask = np.random.rand(pop_size, ND) > cr[:, np.newaxis]
        rows, cols = np.arange(pop_size), np.floor(np.random.rand(pop_size) * ND).astype(int)
        jrand = np.ravel_multi_index((rows, cols), (pop_size, ND), order='F')
        mask.flat[jrand - 1] = False
        ui = vi.copy()
        ui[mask] = pop[mask]
        
        children_fitness = fitness_function(ui)
        print(children_fitness.shape)
        for i in range(pop_size):
            nfes += 1
            if children_fitness[i] < bsf_fit_var:
                bsf_fit_var = children_fitness[i].copy()
                bsf_solution = ui[i, :].copy()
        print(nfes)
        dif = np.abs(fitness - children_fitness)
        I = fitness > children_fitness
        goodCR, goodF, dif_val = cr[I], sf[I], dif[I]
        archive = update_archive(archive, popold[I, :], fitness[I])

        fitness = np.min(np.vstack((fitness, children_fitness)), axis=0)
        popold = pop.copy()
        popold[I, :] = ui[I, :]

        num_success_params = len(goodCR)

        if num_success_params > 0:
            sum_dif = np.sum(dif_val)
            dif_val = dif_val / sum_dif
            memory_sf[memory_pos] = np.dot(dif_val, goodF ** 2) / np.dot(dif_val, goodF)

            if np.max(goodCR) == 0 or memory_cr[memory_pos] == -1:
                memory_cr[memory_pos] = -1
            else:
                memory_cr[memory_pos] = np.dot(dif_val, goodCR ** 2) / np.dot(dif_val, goodCR)

            memory_pos = (memory_pos + 1) % memory_size

        """
            Linear Population size reduction.
        """
        plan_pop_size = round((((min_pop_size - max_pop_size) / max_nfes) * nfes) + max_pop_size)

        if pop_size > plan_pop_size:
            reduction_ind_num = pop_size - plan_pop_size
            if pop_size - reduction_ind_num < min_pop_size:
                reduction_ind_num = pop_size - min_pop_size

            pop_size -= reduction_ind_num

            # Actually delete overflowing particles.
            worst_inds = np.argsort(-fitness)[:reduction_ind_num]
            popold = np.delete(popold, worst_inds, axis=0)
            pop = np.delete(pop, worst_inds, axis=0)
            fitness = np.delete(fitness, worst_inds, axis=0)

            archive['NP'] = int(round(arc_rate * pop_size))

            if archive['pop'].shape[0] > archive['NP']:
                rndpos = np.random.permutation(archive['pop'].shape[0])[:archive['NP']]
                archive['pop'] = archive['pop'][rndpos, :]
        profile.append(copy(bsf_fit_var))

    return np.asarray(profile)

if __name__ == "__main__":
    from hybris.functions import available_functions, test_function
    from hybris.problems import get_benchmark
    from functools import partial

    fitness_function = partial(test_function, name="rastrigin")
    dimensions = 40

    bounds = np.zeros((2, dimensions))
    bounds[0] = - 5.12
    bounds[1] = + 5.12

    bsf_fitness = optimize_lshade(fitness_function, bounds, max_nfes=40000)
    print(bsf_fitness)


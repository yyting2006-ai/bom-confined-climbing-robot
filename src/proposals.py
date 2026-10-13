from pathlib import Path
import runtime

def fixed_pair_proposal(history, seed, pair, excluded=()):
    import numpy as np
    from scipy.stats import qmc
    from scipy.special import ndtr
    from sklearn.gaussian_process import GaussianProcessRegressor
    from sklearn.gaussian_process.kernels import ConstantKernel, Matern
    from search_proposers import checked_history
    if len(pair) != 2 or len(set(pair)) != 2 or any(i not in range(8) for i in pair):
        raise ValueError('Two distinct rod indices required')
    x, y = checked_history(history)
    center = x[int(np.argmax(y))].copy()
    candidates = np.repeat(center[None], 1024, axis=0)
    candidates[:, pair] = .9 + .2*qmc.Sobol(2, scramble=True, seed=seed).random_base2(10)
    gp = GaussianProcessRegressor(kernel=ConstantKernel(1., constant_value_bounds='fixed') *
        Matern(np.full(8, .35), length_scale_bounds='fixed', nu=2.5),
        alpha=1e-4, normalize_y=True, optimizer=None)
    gp.fit((x-.9)/.2, y)
    mu, std = gp.predict((candidates-.9)/.2, return_std=True)
    std = np.maximum(std, 1e-12); delta = mu - np.max(y); z = delta/std
    ei = delta*ndtr(z) + std*np.exp(-.5*z*z)/np.sqrt(2*np.pi)
    blocked = np.concatenate((x, np.asarray(excluded).reshape(-1, 8))) if len(excluded) else x
    distinct = np.min(np.max(np.abs(candidates[:, None]-blocked[None]), axis=2), axis=1) > 1e-7
    if not distinct.any():
        raise ValueError('No distinct fixed-pair candidate')
    ei[~distinct] = -np.inf
    return dict(rod_length_scales=candidates[int(np.argmax(ei))].tolist(), active_rods=list(pair),
                source='GP_EI_fixed_pair', acquisition_value=float(np.max(ei)),
                observations_used=len(history), status='PROPOSED_NOT_EVALUATED')


def proposal(method, observations, step, seed_base, attempt, excluded, feedback, path, scene, pair):
    import numpy as np
    from search_proposers import bom_proposal, text2robot_ea_proposal
    if path.exists():
        return runtime.read(path)
    k = step + 1
    if method in ('bom', 'global_bo', 'random_rod_bo'):
        rod_step = step
        if method == 'random_rod_bo':
            # Same rod for all geometry retries at this step; replacement across steps.
            rod_step = int(np.random.default_rng(seed_base + k).integers(8))
        result = bom_proposal(observations, rod_step, seed_base+k,
                              global_search=method == 'global_bo', excluded=excluded)
        result['selection_rule'] = method
    elif method == 'fixed_pair_2d':
        result = fixed_pair_proposal(observations, seed_base+k, pair, excluded)
    elif method == 'text2robot_ea':
        result = text2robot_ea_proposal(observations, step, seed_base+k*100+attempt)
    elif method == 'robomorph_llm':
        from robomorph_adapter import propose
        return propose(scene, observations, path, constraint_feedback=feedback)
    else:
        raise ValueError(method)
    runtime.save(path, result)
    return result


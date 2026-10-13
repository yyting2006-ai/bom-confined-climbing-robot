"""Candidate rules for the manuscript's fixed-topology task adaptations.

Every fitness input must come from the shared evaluator. These routines do not
train/evaluate controllers or implement the original text-to-3D robot pipeline.
"""
import numpy as np
from scipy.special import ndtr
from scipy.stats import qmc
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel,Matern


def checked_history(history):
    if not history:raise ValueError('A real initial observation is required')
    x=np.asarray([r['rod_length_scales'] for r in history],float)
    y=np.asarray([r['mean_reward'] for r in history],float)
    if x.shape!=(len(history),8) or not np.isfinite(x).all() or not np.isfinite(y).all():
        raise ValueError('Invalid observed data')
    if np.any(x<.9) or np.any(x>1.1):raise ValueError('Observed morphology outside bounds')
    return x,y


def bom_proposal(history,step,seed,global_search=False,excluded=None):
    x,y=checked_history(history)
    if step<0:raise ValueError('Step is zero-based')
    center=x[int(np.argmax(y))].copy()
    active=list(range(8)) if global_search else [step%8]
    unit=qmc.Sobol(d=len(active),scramble=True,seed=seed).random_base2(10)
    candidates=np.repeat(center[None,:],len(unit),axis=0)
    candidates[:,active]=.9+.2*unit
    gp=GaussianProcessRegressor(kernel=ConstantKernel(1.,constant_value_bounds='fixed')*Matern(np.full(8,.35),length_scale_bounds='fixed',nu=2.5),
                                alpha=1e-4,normalize_y=True,optimizer=None)
    gp.fit((x-.9)/.2,y)
    mu,std=gp.predict((candidates-.9)/.2,return_std=True)
    delta=mu-np.max(y);std=np.maximum(std,1e-12);z=delta/std
    acquisition=delta*ndtr(z)+std*np.exp(-.5*z*z)/np.sqrt(2*np.pi)
    distinct=np.min(np.max(np.abs(candidates[:,None,:]-x[None,:,:]),axis=2),axis=1)>1e-7
    if excluded:
        blocked=np.asarray(excluded,float)
        distinct &= np.min(np.max(np.abs(candidates[:,None,:]-blocked[None,:,:]),axis=2),axis=1)>1e-7
    if not distinct.any():raise ValueError('No distinct candidate')
    acquisition[~distinct]=-np.inf
    result=candidates[int(np.argmax(acquisition))]
    return dict(rod_length_scales=result.tolist(),active_rods=active,source='GP_EI',
                observations_used=len(history),acquisition_value=float(np.max(acquisition)),
                status='PROPOSED_NOT_EVALUATED')


def text2robot_ea_proposal(history,step,seed):
    """Small-population, fixed-eight-length EA adaptation described in Appendix A.

    Not official Text2Robot: no text-to-3D, mesh processing, or joint-axis search.
    Initialization, elite count and operator probability are explicit.
    """
    x,y=checked_history(history);rng=np.random.default_rng(seed)
    if len(history)<4:
        candidate=rng.uniform(.9,1.1,8);operator='initial_population_uniform';parents=[]
    else:
        elite=np.argsort(-y,kind='stable')[:4]
        if rng.random()<.5:
            parents=rng.choice(elite,2,replace=False).tolist()
            candidate=np.where(rng.random(8)<.5,x[parents[0]],x[parents[1]])
            operator='uniform_crossover'
        else:
            parents=[int(rng.choice(elite))];candidate=x[parents[0]].copy()
            mask=rng.random(8)<.25
            if not mask.any():mask[int(rng.integers(8))]=True
            candidate[mask]+=rng.normal(0,.025,int(mask.sum()))
            candidate=np.clip(candidate,.9,1.1);operator='gaussian_length_mutation'
    return dict(rod_length_scales=candidate.tolist(),source='Text2Robot_EA_fixed_topology_adaptation',
                operator=operator,parent_observation_indices=parents,
                official_full_text2robot_reproduction=False,status='PROPOSED_NOT_EVALUATED')

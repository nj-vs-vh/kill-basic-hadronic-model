import numpy as np
import emcee
from scipy.optimize import minimize

from dataclasses import dataclass

from nptyping import NDArray
from typing import Optional, List, Any

from . import utils
from .experiment import Object
from .models.base import ModelSED


@dataclass
class ExpSedEnergyVaryingConfig:
    flags: NDArray[(Any,), bool]

    @classmethod
    def for_object(
        cls,
        obj: Object,
        default: bool,
        vary: Optional[List[str]] = None,
        fix: Optional[List[str]] = None,
    ):
        flags = [default] * obj.n_seds
        for polarity, sed_name_part_list in zip([True, False], [vary, fix]):
            if sed_name_part_list is None:
                continue
            for sed_name_part in sed_name_part_list:
                idx = [i for i, sed in enumerate(obj.seds) if sed_name_part in sed.name]
                for i in idx:
                    flags[i] = polarity
        return cls(np.array(flags, dtype=bool))

    def E_factors_from_varying_E_factors(self, varying_E_factors: NDArray[(Any,), bool]):
        E_factors = np.ones((len(self.flags),), dtype=float)
        E_factors[self.flags] = varying_E_factors
        return E_factors

    @property
    def n_varied_E_factors(self) -> int:
        return np.sum(self.flags)


def _theta_to_model_params_and_E_factors(
    theta, model_params_n, exp_E_varying: ExpSedEnergyVaryingConfig
):
    model_params = theta[:model_params_n]
    return tuple(model_params), tuple(
        exp_E_varying.E_factors_from_varying_E_factors(theta[model_params_n:])
    )


def fit_model_to_object(
    obj: Object, model: ModelSED, exp_E_varying: Optional[ExpSedEnergyVaryingConfig] = None
):
    exp_E_varying = exp_E_varying or ExpSedEnergyVaryingConfig.for_object(obj, False)

    print("optimizing model...")

    obj_loglike = obj.get_joint_loglike(model)

    def negloglike(theta):
        return -obj_loglike(
            *_theta_to_model_params_and_E_factors(theta, model.n_params, exp_E_varying)
        )

    # workaround, TODO make estimate_params method of the base class
    model_params_est = (
        np.array(model.estimate_params(obj))
        if hasattr(model, "estimate_params")
        else np.ones((model.n_params,))
    )
    init_pt = np.concatenate((model_params_est, np.ones((exp_E_varying.n_varied_E_factors,))))
    res = minimize(
        negloglike,
        init_pt,
        method="Nelder-Mead",
        tol=1e-7,
        options={"maxfev": 5000000},
    )

    print(f"optimization result: {res.message}")

    if not res.success:
        raise Exception("Optimization didn't succeed :(")

    return _theta_to_model_params_and_E_factors(res.x, model.n_params, exp_E_varying)


@dataclass
class McmcSamplingConfig:
    n_walkers: int = 512
    iterations: int = 5000
    default_tau: int = 300  # tau used when unable to infer from sample


def fit_model_to_object_mcmc(
    obj: Object,
    model: ModelSED,
    exp_E_varying: Optional[ExpSedEnergyVaryingConfig] = None,
    config: Optional[McmcSamplingConfig] = None,
):

    exp_E_varying = exp_E_varying or ExpSedEnergyVaryingConfig.for_object(obj, False)
    config = config or McmcSamplingConfig()

    print("sampling posterior distribution")

    obj_logposterior = obj.get_joint_logposterior(model)

    def logposterior(theta):
        return obj_logposterior(
            *_theta_to_model_params_and_E_factors(theta, model.n_params, exp_E_varying)
        )

    n_dim = model.n_params + exp_E_varying.n_varied_E_factors
    n_walkers = config.n_walkers
    sampler = emcee.EnsembleSampler(n_walkers, n_dim, logposterior)

    theta_estimation = np.array(
        list(model.get_parameters()) + [1.0] * exp_E_varying.n_varied_E_factors
    )
    theta_init_log_sigmas = np.concatenate(
        (0.1 * np.ones((model.n_params)), 0.05 * np.ones((exp_E_varying.n_varied_E_factors)))
    )

    starting_points = np.tile(theta_estimation, (n_walkers, 1))
    for i, sigma in enumerate(theta_init_log_sigmas):
        starting_points[:, i] = np.exp(
            np.random.normal(loc=0, scale=np.log(1 + sigma), size=(n_walkers,))
        )

    sampler.run_mcmc(starting_points, config.iterations, progress=True)

    print(f"acc. frac. = {np.mean(sampler.acceptance_fraction)} ([0.2; 0.5] range is expected)")

    taus = sampler.get_autocorr_time(quiet=True)
    if np.all(np.isnan(taus)):
        print("unable to calculate tau, using given default value")
        tau = config.default_tau
    else:
        tau = int(np.max(taus[np.isfinite(taus)]))
    print(f"tau = {tau}")

    sample = sampler.get_chain(flat=True, thin=tau, discard=tau * 10)
    print(f"got {sample.shape[0]} samples")

    print("looking for max-likelihood point in sample")

    obj_loglike = obj.get_joint_loglike(model)

    def loglike(theta):
        return obj_loglike(
            *_theta_to_model_params_and_E_factors(theta, model.n_params, exp_E_varying)
        )

    theta_maxloglike = utils.max_loglike_point(sample, loglike, progress=True)
    return _theta_to_model_params_and_E_factors(theta_maxloglike, model.n_params, exp_E_varying)

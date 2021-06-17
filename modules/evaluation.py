import numpy as np

from typing import List, Union

from .utils import AIC
from .experiment import Object
from .models.base import ModelSED
from .fitting import ExpSedEnergyVaryingConfig


def evaluate_model_for_object(
    obj: Object, model: ModelSED, exp_E_varying: ExpSedEnergyVaryingConfig
):
    obj_loglike = obj.get_joint_loglike(model)
    maxloglike = obj_loglike(tuple(model.get_parameters()), tuple(np.ones((obj.n_seds))))
    return AIC(k=model.n_params + exp_E_varying.n_varied_E_factors, max_loglikelihood=maxloglike)


def compare_models_for_objects(
    objects: Union[Object, List[Object]],
    models: List[ModelSED],
    exp_E_varying_configs: List[ExpSedEnergyVaryingConfig],
):
    if isinstance(objects, Object):
        objects = [objects] * len(models)

    aic_scores = []
    for obj, model, exp_E_var_conf in zip(objects, models, exp_E_varying_configs):
        aic_scores.append(evaluate_model_for_object(obj, model, exp_E_var_conf))

    sort_res = sorted(zip(aic_scores, models, objects), key=lambda aic, *_: aic)
    aic_scores = [sr[0] for sr in sort_res]
    models = [sr[1] for sr in sort_res]
    objects = [sr[2] for sr in sort_res]

    delta_aic = [aic - aic_scores[0] for aic in aic_scores]
    likelihoods = np.array([np.exp(-0.5 * daic) for daic in delta_aic])
    akaike_weights = likelihoods / np.sum(likelihoods)

    def results_repr(weight, model, obj) -> str:
        return f"\t{model}\t|\t{obj}\t|\t{weight:.6f}"

    header = "\tModel\t|\tObject\t|\tWeight"

    newline = "\n"

    print(
        f"Akaike weights:\n\n{header}\n{newline.join(results_repr(*t) for t in zip(akaike_weights, models, objects))}"
    )

    return akaike_weights

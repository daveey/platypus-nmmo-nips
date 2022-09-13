import collections

import torch
from torch import Tensor

GAEReturns = collections.namedtuple("GAEReturns", "vs advantages")


@torch.no_grad()
def compute_return_and_advantage(
    value: Tensor,
    reward: Tensor,
    bootstrap_value: Tensor,
    discount: Tensor,  # (1 - done) * gamma
    lambda_: float = 1.0,
    mask: Tensor = None,
):
    """
    value: [T, ...]
    reward: [T, ...]
    bootstrap_value: [...]
    discount: [T, ...]
    """
    T = value.shape[0]
    value = torch.cat([value, bootstrap_value.unsqueeze(dim=0)], dim=0)
    delta = reward + discount * value[1:] - value[:-1]
    last_gae_lam = torch.zeros_like(bootstrap_value)
    result = []
    for t in reversed(range(T)):
        last_gae_lam = delta[t] + discount[t] * lambda_ * last_gae_lam
        result.append(last_gae_lam)
    result.reverse()
    adv = torch.stack(result)
    return_ = adv + value[:-1]
    if mask is not None:
        adv *= mask
        return_ *= mask
    return GAEReturns(vs=return_, advantages=adv)

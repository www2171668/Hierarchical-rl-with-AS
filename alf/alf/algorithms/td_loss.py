

import torch
import torch.nn as nn

import alf
from alf.data_structures import LossInfo, StepType
from alf.utils.losses import element_wise_squared_loss
from alf.utils import tensor_utils, value_ops
from alf.utils.summary_utils import safe_mean_hist_summary
from alf.utils.normalizers import AdaptiveNormalizer


@alf.configurable
class TDLoss(nn.Module):
    """Temporal difference loss."""

    def __init__(self,
                 gamma=0.98,
                 td_error_loss_fn=element_wise_squared_loss,
                 td_lambda=0.95,
                 normalize_target=False,
                 debug_summaries=False,
                 name="TDLoss"):
        r"""
        Let :math:`G_{t:T}` be the bootstraped return from t to T:

        .. math::
          G_{t:T} = \sum_{i=t+1}^T \gamma^{t-i-1}R_i + \gamma^{T-t} V(s_T)

        If ``td_lambda`` = 1, the target for step t is :math:`G_{t:T}`.
        If ``td_lambda`` = 0, the target for step t is :math:`G_{t:t+1}`
        If 0 < ``td_lambda`` < 1, the target for step t is the :math:`\lambda`-return:

        .. math::
            G_t^\lambda = (1 - \lambda) \sum_{i=t+1}^{T-1} \lambda^{i-t}G_{t:i} + \lambda^{T-t-1} G_{t:T}

        There is a simple relationship between :math:`\lambda`-return and
        the generalized advantage estimation :math:`\hat{A}^{GAE}_t`:

        .. math::
            G_t^\lambda = \hat{A}^{GAE}_t + V(s_t)

        where the generalized advantage estimation is defined as:

        .. math::
            \hat{A}^{GAE}_t = \sum_{i=t}^{T-1}(\gamma\lambda)^{i-t}(R_{i+1} + \gamma V(s_{i+1}) - V(s_i))

        Args:
            gamma (float|list[float]): A discount factor for future rewards. For
                multi-dim reward, this can also be a list of discounts, each
                discount applies to a reward dim.
            td_errors_loss_fn (Callable): A function for computing the TD errors
                loss. This function takes as input the target and the estimated
                Q values and returns the loss for each element of the batch.
            td_lambda (float): Lambda parameter for TD-lambda computation.
            normalize_target (bool): whether to normalize target.
                Note that the effect of this is to change the loss. The critic
                value itself is not normalized.
            debug_summaries (bool): True if debug summaries should be created.
            name (str): The name of this loss.
        """
        super().__init__()

        self._name = name
        self._gamma = torch.tensor(gamma)
        self._td_error_loss_fn = td_error_loss_fn
        self._lambda = td_lambda
        self._debug_summaries = debug_summaries
        self._normalize_target = normalize_target
        self._target_normalizer = None

    @property
    def gamma(self):
        """Return the :math:`\gamma` value for discounting future rewards.

        Returns:
            Tensor: a rank-0 or rank-1 (multi-dim reward) floating tensor.
        """
        return self._gamma.clone()

    def forward(self, info, value, target_value):
        """Calculate the loss.

        The first dimension of all the tensors is time dimension and the second
        dimesion is the batch dimension.

        Args:
            info (namedtuple): experience collected from ``unroll()`` or
                a replay buffer. All tensors are time-major. ``info`` should
                contain the following fields:
                - reward:
                - step_type:
                - discount:
            value (torch.Tensor): the time-major tensor for the value at each time
                step. The loss is between this and the calculated return.
            target_value (torch.Tensor): the time-major tensor for the value at
                each time step. This is used to calculate return. ``target_value``
                can be same as ``value``.
        Returns:
            LossInfo: with the ``extra`` field same as ``loss``.
        """
        if info.reward.ndim == 3:
            discounts = info.discount.unsqueeze(-1) * self._gamma
        else:
            discounts = info.discount * self._gamma

        if self._lambda == 1.0:
            returns = value_ops.discounted_return(
                rewards=info.reward,
                values=target_value,
                step_types=info.step_type,
                discounts=discounts)
        elif self._lambda == 0.0:
            returns = value_ops.one_step_discounted_return(
                rewards=info.reward,
                values=target_value,
                step_types=info.step_type,
                discounts=discounts)
        else:
            advantages = value_ops.generalized_advantage_estimation(
                rewards=info.reward,
                values=target_value,
                step_types=info.step_type,
                discounts=discounts,
                td_lambda=self._lambda)
            returns = advantages + target_value[:-1]

        value = value[:-1]
        if self._normalize_target:
            if self._target_normalizer is None:
                self._target_normalizer = AdaptiveNormalizer(
                    alf.TensorSpec(value.shape[2:]),
                    auto_update=False,
                    debug_summaries=self._debug_summaries,
                    name=self._name + ".target_normalizer")

            self._target_normalizer.update(returns)
            returns = self._target_normalizer.normalize(returns)
            value = self._target_normalizer.normalize(value)

        loss = self._td_error_loss_fn(returns.detach(), value)

        if loss.ndim == 3:
            # Multidimensional reward. Average over the critic loss for all dimensions
            loss = loss.mean(dim=2)

        # The shape of the loss expected by Algorith.update_with_gradient is
        # [T, B], so we need to augment it with additional zeros.
        loss = tensor_utils.tensor_extend_zero(loss)
        return LossInfo(loss=loss, extra=loss)

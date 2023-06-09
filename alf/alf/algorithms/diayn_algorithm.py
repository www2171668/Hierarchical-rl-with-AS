

import torch

import alf
from alf.algorithms.algorithm import Algorithm
from alf.data_structures import AlgStep, LossInfo, namedtuple, TimeStep, StepType
from alf.networks import EncodingNetwork
from alf.tensor_specs import BoundedTensorSpec, TensorSpec
from alf.utils.tensor_utils import to_tensor
from alf.utils import math_ops
from alf.utils.normalizers import AdaptiveNormalizer, ScalarAdaptiveNormalizer

DIAYNInfo = namedtuple("DIAYNInfo", ["loss"])

# % 连续技能
@alf.configurable
def create_skill_spec(num_of_skills):
    return BoundedTensorSpec((num_of_skills,), maximum=1, minimum=-1)

# % 离散技能
@alf.configurable
def create_discrete_skill_spec(num_of_skills):
    return BoundedTensorSpec((), dtype="int64", maximum=num_of_skills - 1)

@alf.configurable
class DIAYNAlgorithm(Algorithm):
    """Diversity is All You Need Module

    This module learns a set of skill-conditional policies in an unsupervised
    way. See Eysenbach et al "Diversity is All You Need: Learning Diverse Skills
    without a Reward Function" for more details.
    """

    def __init__(self,
                 skill_spec,
                 encoding_net: EncodingNetwork,
                 reward_adapt_speed=8.0,
                 observation_spec=None,
                 hidden_size=(),
                 hidden_activation=torch.relu_,
                 name="DIAYNAlgorithm"):
        """Create a DIAYNAlgorithm.

        Args:
            skill_spec (TensorSpec): supports both discrete and continuous skills.
                In the discrete case, the algorithm will predict 1-of-K skills
                using the cross entropy loss; in the continuous case, the
                algorithm will predict the skill vector itself using the mean
                square error loss.
            encoding_net (EncodingNetwork): network for encoding observation into
                a latent feature.
            reward_adapt_speed (float): how fast to adapt the reward normalizer.
                rouphly speaking, the statistics for the normalization is
                calculated mostly based on the most recent `T/speed` samples,
                where `T` is the total number of samples.
            observation_spec (TensorSpec): If not None, this spec is to be used
                by a observation normalizer to normalize incoming observations.
                In some cases, the normalized observation can be easier for
                training the discriminator.
            hidden_size (tuple[int]): a tuple of hidden layer sizes used by the
                discriminator.
            hidden_activation (torch.nn.functional): activation for the hidden
                layers.
            name (str): module's name
        """
        assert isinstance(skill_spec, TensorSpec)

        self._skill_spec = skill_spec
        if skill_spec.is_discrete:
            assert isinstance(skill_spec, BoundedTensorSpec)
            skill_dim = skill_spec.maximum - skill_spec.minimum + 1
        else:
            assert len(skill_spec.shape) == 1, "Only 1D skill vector is supported"
            skill_dim = skill_spec.shape[0]

        super().__init__(
            train_state_spec=TensorSpec((skill_dim, )),
            predict_state_spec=(),  # won't be needed for predict_step
            name=name)

        # % 编码器
        self._encoding_net = encoding_net  # s->z

        # % 判别器
        self._discriminator_net = EncodingNetwork(
            input_tensor_spec=encoding_net.output_spec,
            fc_layer_params=hidden_size,
            activation=hidden_activation,
            last_layer_size=skill_dim,  # 预测skill
            last_activation=math_ops.identity)

        # % 定义正则化器
        self._reward_normalizer = ScalarAdaptiveNormalizer(
            speed=reward_adapt_speed)

        self._observation_normalizer = None
        if observation_spec is not None:
            self._observation_normalizer = AdaptiveNormalizer(
                tensor_spec=observation_spec)

    def _step(self, time_step: TimeStep, state, calc_rewards=True):
        """
        Args:
            time_step (TimeStep): input time step data, where the
                observation is skill-augmened observation. The skill should be
                a one-hot vector.
            state (Tensor): state for DIAYN (previous skill) which should be
                a one-hot vector.
            calc_rewards (bool): if False, only return the losses.

        Returns:
            AlgStep:
                output: empty tuple ()
                state: skill
                info (DIAYNInfo):
        """
        observations_aug = time_step.observation
        step_type = time_step.step_type
        observation, skill = observations_aug  # 增广状态
        prev_skill = state.detach()  # 技能z

        # % normalize observation for easier prediction
        if self._observation_normalizer is not None:
            observation = self._observation_normalizer.normalize(observation)

        # % 编码器 s->pred_z
        if self._encoding_net is not None:
            feature, _ = self._encoding_net(observation)

        skill_pred, _ = self._discriminator_net(feature)

        # % 计算损失
        if self._skill_spec.is_discrete:
            loss = torch.nn.CrossEntropyLoss(reduction='none')(
                input=skill_pred, target=torch.argmax(prev_skill, dim=-1))
        else:
            loss = torch.sum(math_ops.square(skill_pred - prev_skill), dim=-1)

        valid_masks = (step_type != to_tensor(StepType.FIRST)).to(torch.float32)  # FIRST不计算loss
        loss *= valid_masks

        # % 计算内部奖励
        intrinsic_reward = ()
        if calc_rewards:
            intrinsic_reward = -loss.detach()
            intrinsic_reward = self._reward_normalizer.normalize(intrinsic_reward)

        return AlgStep(
            output=intrinsic_reward, state=skill, info=DIAYNInfo(loss=loss))

    def rollout_step(self, inputs, state):
        """获得内部奖励和损失"""
        return self._step(inputs, state)

    def train_step(self, inputs, state, rollout_info=None):
        """获得损失"""
        return self._step(inputs, state, calc_rewards=False)  # 训练时不再计算内部奖励,只需要loss

    def calc_loss(self, info: DIAYNInfo):
        """损失求平均"""
        loss = torch.mean(info.loss)
        return LossInfo(scalar_loss=loss, extra=dict(skill_discriminate_loss=info.loss))

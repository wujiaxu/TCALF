import torch
from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler

from TCALF.url_benchmark.custom_dmc_tasks.quadruped import Roll


def _flatten_helper(T, N, _tensor):
    if isinstance(_tensor, dict):
        for key in _tensor:
            _tensor[key] = _tensor[key].view(T * N, *(_tensor[key].size()[2:]))
        return _tensor
    else:
        return _tensor.view(T * N, *_tensor.size()[2:])

class RolloutStorage(object):
    """ The rollout buffer to store the agent's experience for PPO """
    def __init__(self, num_steps, human_num, obs_shape, action_space, rnn_size, meta_size, discount):

        self._discount = discount
        self.obs = torch.zeros(num_steps + 1, human_num, *obs_shape)
        self.recurrent_hidden_states = torch.zeros(num_steps + 1, human_num, rnn_size)
        self.rewards = torch.zeros(num_steps, human_num, 1)
        self.value_preds = torch.zeros(num_steps + 1, human_num, 1)
        self.returns = torch.zeros(num_steps + 1, human_num, 1)
        self.action_log_probs = torch.zeros(num_steps, human_num, 1)
        action_shape = action_space.shape[0]
        self.actions = torch.zeros(num_steps, human_num, action_shape)
        self.masks = torch.ones(num_steps + 1, human_num, 1)
        self.metas = torch.zeros(num_steps,human_num,meta_size)

        self.num_steps = num_steps
        self.num_agent = human_num
        self.step = 0

    def to(self, device):
        self.obs = self.obs.to(device)
        self.recurrent_hidden_states= self.recurrent_hidden_states.to(device)

        self.rewards = self.rewards.to(device)
        self.value_preds = self.value_preds.to(device)
        self.returns = self.returns.to(device)
        self.action_log_probs = self.action_log_probs.to(device)
        self.actions = self.actions.to(device)
        self.masks = self.masks.to(device)
        self.metas = self.metas.to(device)

    def insert(self, obs, recurrent_hidden_states, metas, actions, action_log_probs,
               value_preds, rewards, masks):


        self.obs[self.step + 1].copy_(obs)
        self.recurrent_hidden_states[self.step + 1].copy_(recurrent_hidden_states)

        self.actions[self.step].copy_(actions)
        self.action_log_probs[self.step].copy_(action_log_probs)
        self.value_preds[self.step].copy_(value_preds)
        self.rewards[self.step].copy_(rewards)
        self.masks[self.step + 1].copy_(masks)
        self.metas[self.step + 1].copy_(metas)

        self.step = (self.step + 1) % self.num_steps

    def after_update(self):

        self.obs[0].copy_(self.obs[-1])
        self.recurrent_hidden_states[0].copy_(self.recurrent_hidden_states[-1])

        self.masks[0].copy_(self.masks[-1])
        self.metas[0].copy_(self.metas[-1])

    def compute_returns(self,
                        next_value,
                        use_gae,
                        gae_lambda):
        if use_gae:
            self.value_preds[-1] = next_value
            gae = 0
            for step in reversed(range(self.rewards.size(0))):
                delta = self.rewards[step] + self._discount * self.value_preds[
                    step + 1] * self.masks[step +
                                            1] - self.value_preds[step]
                gae = delta + self._discount * gae_lambda * self.masks[step +
                                                                1] * gae
                self.returns[step] = gae + self.value_preds[step]
        else:
            self.returns[-1] = next_value
            for step in reversed(range(self.rewards.size(0))):
                self.returns[step] = self.returns[step + 1] * \
                    self._discount * self.masks[step + 1] + self.rewards[step]

    def recurrent_generator(self, advantages, num_mini_batch):
        
        assert self.num_agent >= num_mini_batch, (
            "PPO requires the number of processes ({}) "
            "to be greater than or equal to the number of "
            "PPO mini batches ({}).".format(self.num_agent, num_mini_batch))
        num_agent_per_batch = self.num_agent // num_mini_batch
        perm = torch.randperm(self.num_agent)
        for start_ind in range(0, self.num_agent, num_agent_per_batch):

            obs_batch = []
            recurrent_hidden_states_batch = []
            actions_batch = []
            value_preds_batch = []
            return_batch = []
            masks_batch = []
            metas_batch = []
            old_action_log_probs_batch = []
            adv_targ = []

            for offset in range(num_agent_per_batch):
                ind = perm[start_ind + offset]

                obs_batch.append(self.obs[:-1, ind])
                recurrent_hidden_states_batch.append(self.recurrent_hidden_states[0:1, ind])

                actions_batch.append(self.actions[:, ind])
                metas_batch.append(self.metas[:, ind])
                value_preds_batch.append(self.value_preds[:-1, ind])
                return_batch.append(self.returns[:-1, ind])
                masks_batch.append(self.masks[:-1, ind])
                old_action_log_probs_batch.append(
                    self.action_log_probs[:, ind])
                adv_targ.append(advantages[:, ind])

            T, N = self.num_steps, num_agent_per_batch
            # These are all tensors of size (T, N, -1)

            actions_batch = torch.stack(actions_batch, 1)
            metas_batch = torch.stack(metas_batch,1)
            value_preds_batch = torch.stack(value_preds_batch, 1)
            return_batch = torch.stack(return_batch, 1)
            masks_batch = torch.stack(masks_batch, 1)
            old_action_log_probs_batch = torch.stack(
                old_action_log_probs_batch, 1)
            adv_targ = torch.stack(adv_targ, 1)

            obs_batch = torch.stack(obs_batch, 1)
            temp = torch.stack(recurrent_hidden_states_batch, 1)
            recurrent_hidden_states_batch= temp.view(N, *(temp.size()[2:]))

            # Flatten the (T, N, ...) tensors to (T * N, ...)
            obs_batch = _flatten_helper(T, N, obs_batch)
            actions_batch = _flatten_helper(T, N, actions_batch)
            metas_batch = _flatten_helper(T,N,metas_batch)
            value_preds_batch = _flatten_helper(T, N, value_preds_batch)
            return_batch = _flatten_helper(T, N, return_batch)
            masks_batch = _flatten_helper(T, N, masks_batch)
            old_action_log_probs_batch = _flatten_helper(T, N, \
                    old_action_log_probs_batch)
            adv_targ = _flatten_helper(T, N, adv_targ)

            yield obs_batch, recurrent_hidden_states_batch, actions_batch, metas_batch,\
                value_preds_batch, return_batch, masks_batch, old_action_log_probs_batch, adv_targ

class DiscrimRolloutStorage(RolloutStorage):
    def __init__(self, discrim_fn, discrim_weight, num_steps, human_num, obs_shape, action_space, rnn_size, meta_size, discount):
        super().__init__(num_steps, human_num, obs_shape, action_space, rnn_size, meta_size, discount)
        self._discrim_fn = discrim_fn
        self._discrim_weight = discrim_weight
    
    def compute_returns(self,
                        next_value,
                        use_gae,
                        gae_lambda):
        """
        Adds a weighted diversity reward to the task reward in the behavior
        agent's rollout buffer. This overrides the existing rewards in the
        buffer. The buffer of the coordination agent is unmodified.
        """
        #TODO
        # behav_storage = self._active_storages[BEHAV_AGENT]
        # with torch.no_grad():
        #     masks = behav_storage.buffers["masks"]
        #     obs = behav_storage.buffers["observations"]
        #     features, _ = self._hl_policy(
        #         obs.map(lambda x: x.flatten(0, 1)),
        #         behav_storage.buffers["recurrent_hidden_states"].flatten(0, 1),
        #         masks.flatten(0, 1),
        #     )
        #     features = features.view(*masks.shape[:2], -1)
        #     pred_logits = self._discrim.pred_logits(features, obs)
        #     behav_ids = torch.argmax(obs[BEHAV_ID], -1).long()
        #     scores = F.log_softmax(pred_logits, -1)
        #     log_prob = scores.gather(-1, behav_ids.view(*masks.shape[:2], 1))
        #     behav_storage.buffers["rewards"] += (
        #         self._discrim_reward_weight * log_prob
        #     )
        
        current_hidden_states = self.recurrent_hidden_states[-1] # N X H
        metas = self.metas[-1] # N X C
        with torch.no_grad():
            intrin_reward = torch.einsum("bi,bi->b", metas, self._discrim_fn(current_hidden_states, norm=True))*self._discrim_weight
        self.rewards[-1] += intrin_reward.unsqueeze(-1)

        super().compute_returns(next_value,
                        use_gae,
                        gae_lambda)
from url_benchmark.crowd_sim.utils.agent import Agent
from url_benchmark.crowd_sim.utils.state import JointState
from url_benchmark.crowd_sim import policy
from url_benchmark.crowd_sim.utils.action import ActionVW


class Robot(Agent):
    def __init__(self, config, section):
        super().__init__(config, section)
        self.rotation_constraint = getattr(config, section).rotation_constraint

    def act(self, ob):
        if self.policy is None:
            raise AttributeError('Policy attribute has to be set!')
        state = JointState(self.get_full_state(), ob)
        state.w = self.w
        action, action_index = self.policy.predict(state)
        if isinstance(action, ActionVW):
            self.w = action.w
        return action, action_index

    def get_state(self, ob):
        state = JointState(self.get_full_state(), ob)
        if self.policy is None:
            raise AttributeError('Policy attribute has to be set!')
        return self.policy.transform(state)

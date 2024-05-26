from controllable_navi.agent.GD_aps import *

@dataclasses.dataclass
class GD_RNN_APSAgentConfig(GD_APSAgentConfig):
    _target_: str = "controllable_navi.agent.GD_rnn_aps.RNN_APSAgent" 
    name: str = "gd_rnn_aps"

cs = ConfigStore.instance()
cs.store(group="agent", name="gd_rnn_aps", node=GD_RNN_APSAgentConfig)

class RNN_APSAgent(APSAgent):

    def __init__(self, **kwargs: tp.Any) -> None:
        cfg = GD_RNN_APSAgentConfig(**kwargs)

        # create actor and critic
        # increase obs shape to include task dim (through meta_dim)
        super().__init__(**kwargs)
        self.cfg: GD_RNN_APSAgentConfig = cfg  # override base ddpg cfg type
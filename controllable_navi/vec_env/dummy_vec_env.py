import numpy as np
from controllable_navi.vec_env.vec_env import VecEnv
# from controllable_navi.vec_env.util import dict_to_obs, obs_space_info, copy_obs_dict
from controllable_navi.crowd_sim.crowd_sim import NaviObsSpace
from controllable_navi.crowd_sim.crowd_sim import InformedTimeStep
from controllable_navi.dm_env_light.specs import BoundedArray

class DummyVecEnv(VecEnv):
    """
    VecEnv that does runs multiple environments sequentially, that is,
    the step and reset commands are send to one environment at a time.
    Useful when debugging and when num_env == 1 (in the latter case,
    avoids communication overhead)
    """
    def __init__(self, env_fns,human_nums):
        """
        Arguments:

        env_fns: iterable of callables      functions that build environments
        """
        self.envs = [fn() for fn in env_fns]
        total_agent_num = sum(human_nums)
        env = self.envs[0]
        VecEnv.__init__(self, len(env_fns), env.observation_spec(), env.action_spec())
        obs_space = env.observation_spec()
        # self.keys, shapes, dtypes = obs_space_info(obs_space) #TODO
        self.obs_shape = obs_space.shape
        self.obs_dtype = obs_space.dtype

        self.buf_obs = np.zeros((total_agent_num,) + self.obs_shape, dtype=self.obs_dtype)
        self.buf_discount = np.zeros((total_agent_num,), dtype=np.float32)
        self.buf_rews  = np.zeros((total_agent_num,), dtype=np.float32)
        self.buf_infos = [] #TODO
        self.buf_physics = []
        self.actions = None
        # self.spec = self.envs[0].spec #TODO ?

    def step_async(self, actions):
        listify = True
        try:
            if len(actions) == self.num_envs:
                listify = False
        except TypeError:
            pass

        if not listify:
            self.actions = actions
        else:
            assert self.num_envs == 1, "actions {} is either not a list or has a wrong size - cannot match to {} environments".format(actions, self.num_envs)
            self.actions = [actions]

    def step_wait(self):
        human_id = 0
        self.buf_infos = [] #TODO
        self.buf_physics = []
        for e in range(self.num_envs):
            action = self.actions[e]
            # if isinstance(self.envs[e].action_space, spaces.Discrete):
            #    action = int(action)
            #self.buf_obs[e], self.buf_rews[e], self.buf_dones[e], self.buf_infos[e] = self.envs[e].step(action) #TODO
            
            multi_agent_step = self.envs[e].step(action)
            for agent_step in multi_agent_step:
                self.buf_obs[human_id]= agent_step.observation
                self.buf_rews[human_id]=agent_step.reward
                self.buf_discount[human_id]=agent_step.discount
                self.buf_infos.append(agent_step.info)
                self.buf_physics.append(agent_step.physics)
                human_id += 1
            # if self.buf_dones[e]: #TODO
            #     obs = self.envs[e].reset()
            # self._save_obs(e, obs)
        # return (self._obs_from_buf(), np.copy(self.buf_rews), np.copy(self.buf_discount),
        #         self.buf_infos.copy(),self.buf_physics.copy())
        return (np.copy(self.buf_obs), np.copy(self.buf_rews), np.copy(self.buf_discount),
                self.buf_infos.copy(),self.buf_physics.copy())

    def reset(self):
        # for e in range(self.num_envs):
        #     obs = self.envs[e].reset()
        #     self._save_obs(e, obs)
        human_id = 0

        for e in range(self.num_envs):
            multi_agent_step = self.envs[e].reset()
            for agent_step in multi_agent_step:
                self.buf_obs[human_id]= agent_step.observation
                human_id += 1
        # return self._obs_from_buf()
        return np.copy(self.buf_obs)

    def talk2Env_async(self, data):
        self.envs[0].env.talk2Env(data[0])
        pass

    def talk2Env_wait(self):
        return [True]

    # def _save_obs(self, e, obs):
    #     for k in self.keys:
    #         if k is None:
    #             self.buf_obs[k][e] = obs
    #         else:
    #             self.buf_obs[k][e] = obs[k]

    # def _obs_from_buf(self):
    #     return dict_to_obs(copy_obs_dict(self.buf_obs)) #TODO

    def get_images(self):
        return [env.render(return_rgb=True) for env in self.envs]

    def render(self, mode='rgb_array'):
        if self.num_envs == 1:
            if mode=='human':
                # return self.envs[0].render(return_rgb=False)
                raise NotImplementedError
            elif mode=='rgb_array':
                return self.envs[0].render(return_rgb=True)
            else:
                raise NotImplementedError
        else:
            return self.get_images()

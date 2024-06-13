"""
An interface for asynchronous vectorized environments.
"""

import dis
import multiprocessing as mp
import numpy as np
from controllable_navi.vec_env.vec_env import VecEnv, CloudpickleWrapper, clear_mpi_env_vars
import ctypes
import typing as tp
# from controllable_navi.vec_env.util import dict_to_obs, obs_space_info, obs_to_dict
from controllable_navi.crowd_sim.crowd_sim import NaviObsSpace
from controllable_navi.crowd_sim.crowd_sim import InformedTimeStep
from controllable_navi.dm_env_light.specs import BoundedArray

_NP_TO_CT = {
             np.float32: ctypes.c_float,
             np.int32: ctypes.c_int32,
             np.int8: ctypes.c_int8,
             np.uint8: ctypes.c_char,
             bool: ctypes.c_bool,
             np.bool_: ctypes.c_bool}


class ShmemVecEnv(VecEnv):
    """
    Optimized version of SubprocVecEnv that uses shared variables to communicate observations.
    """

    def __init__(self, env_fns, human_nums, spaces=None, context='spawn'):
        """
        If you don't specify observation_space, we'll have to create a dummy
        environment to get it.
        """
        ctx = mp.get_context(context)
        observation_space:NaviObsSpace
        action_space:BoundedArray

        if spaces:
            observation_space, action_space = spaces
        else:
            # with logger.scoped_configure(format_strs=[]):
            dummy = env_fns[0]()
            observation_space, action_space = dummy.observation_spec(), dummy.action_spec()
            dummy.close()
            del dummy
        VecEnv.__init__(self, len(env_fns), observation_space, action_space)
        # self.obs_keys, self.obs_shapes, self.obs_dtypes = obs_space_info(observation_space) #TODO
        self.obs_shape = observation_space.shape
        self.obs_dtype = observation_space.dtype
        
        self.obs_bufs = [
            [ctx.Array(ctypes.c_float, int(np.prod(self.obs_shape))) 
             for _ in range(human_num)]
            for human_num in human_nums]
        self.phy_bufs = [
            [ctx.Array(ctypes.c_float, int(1+human_num*5+100)) 
             for _ in range(human_num)]
            for human_num in human_nums]
        self.phy_shapes = [int(1+human_num*5+100) for human_num in human_nums]
        self.parent_pipes = []
        self.procs = []
        with clear_mpi_env_vars():
            for env_fn, obs_buf,phy_bufs,phy_shape in zip(env_fns, self.obs_bufs,self.phy_bufs,self.phy_shapes):
                wrapped_fn = CloudpickleWrapper(env_fn)
                parent_pipe, child_pipe = ctx.Pipe()
                proc = ctx.Process(target=_subproc_worker,
                            args=(child_pipe, 
                                  parent_pipe, 
                                  wrapped_fn, 
                                  obs_buf, 
                                  phy_bufs,
                                  phy_shape,
                                  self.obs_shape, 
                                  self.obs_dtype))
                proc.daemon = True
                self.procs.append(proc)
                self.parent_pipes.append(parent_pipe)
                proc.start()
                child_pipe.close()
        self.waiting_step = False
        self.viewer = None

    def reset(self):
        if self.waiting_step:
            self.step_wait()
        for pipe in self.parent_pipes:
            pipe.send(('reset', None))
        obs = []
        for pipe in self.parent_pipes:
            o = pipe.recv()
            obs.append(o)
        return self._decode_obses(obs)

    def step_async(self, actions):
        assert len(actions) == len(self.parent_pipes)
        for pipe, act in zip(self.parent_pipes, actions):
            pipe.send(('step', act))
        self.waiting_step = True

    def step_wait(self):
        # outs = [pipe.recv() for pipe in self.parent_pipes]
        # self.waiting_step = False
        # need to concatenate along human/robot axis
        # obs, rews, dones, infos = zip(*outs)
        obs = []
        rews = []
        discounts = []
        infos = []
        phy = []
        for pipe in self.parent_pipes:
            o, r, discount, inf,p = pipe.recv()
            obs.append(o)
            rews+=r
            discounts+=discount
            infos+=inf
            phy.append(p)
        self.waiting_step = False
        obs_all = self._decode_obses(obs)
        phy_all = self._decode_phy(phy)
        return obs_all, np.array(rews), np.array(discounts), infos, phy_all

    def talk2Env_async(self, data):
        assert len(data) == len(self.parent_pipes)
        for pipe, d in zip(self.parent_pipes, data):
            pipe.send(('talk2Env', d))
        self.waiting_step = True

    def talk2Env_wait(self):
        outs = [pipe.recv() for pipe in self.parent_pipes]  # pipe.recv() is a blocking call
        self.waiting_step = False
        return np.array(outs)

    def close_extras(self):
        if self.waiting_step:
            self.step_wait()
        for pipe in self.parent_pipes:
            pipe.send(('close', None))
        for pipe in self.parent_pipes:
            pipe.recv()
            pipe.close()
        for proc in self.procs:
            proc.join()

    # def render(self, mode='human'):
    #     if self.num_envs == 1:
    #         if mode=='human':
    #             return self.envs[0].render(return_rgb=False)
    #         elif mode=='rgb_array':
    #             return self.envs[0].render(return_rgb=True)
    #         else:
    #             raise NotImplementedError
    #     else:
    #         return super().render(mode=mode)

    def get_images(self, mode='human'):
        for pipe in self.parent_pipes:
            pipe.send(('render', None))
        return [pipe.recv() for pipe in self.parent_pipes]

    def _decode_obses(self, all_pipe_obs):
        """
        obs : [N X obs_shape, M X obs_shape,..., L X obs_shape]
        """
        # result = {}
        # for k in self.obs_keys:

        #     bufs = [b[k] for b in self.obs_bufs]
        #     o = [np.frombuffer(
        #             b.get_obj(), 
        #             dtype=self.obs_dtypes[k]).reshape(self.obs_shapes[k]) \
        #         for b in bufs]
        #     result[k] = np.array(o)

        # return dict_to_obs(result) #TODO
        o = []
        for buf in self.obs_bufs:
            for agent_buf in buf:
                o.append(np.frombuffer(
                    agent_buf.get_obj(), 
                    dtype=self.obs_dtype).reshape((1,)+self.obs_shape) )

        return np.concatenate(o,axis=0)
    
    def _decode_phy(self,all_pipe_phy):
        o = []
        for i, buf in enumerate(self.phy_bufs):
            for agent_buf in buf:
                data = np.frombuffer(
                    agent_buf.get_obj(), 
                    dtype=self.obs_dtype).reshape(1,self.phy_shapes[i])
                o.append(data)

        return o


def _subproc_worker(pipe, parent_pipe, env_fn_wrapper, obs_bufs, phy_bufs, phy_shape, obs_shape, obs_dtype):
    """
    Control a single environment instance using IPC and
    shared memory.
    """
    def _write_obs(multi_agent_obs_list):
        """
        consider each subobservation space is concatnated
        """
        # flatdict = obs_to_dict(maybe_dict_obs) #TODO
        # for k in keys:
        #     dst = obs_bufs[k].get_obj()
        #     dst_np = np.frombuffer(dst, dtype=obs_dtypes[k]).reshape(obs_shapes[k])  # pylint: disable=W0212
        #     np.copyto(dst_np, flatdict[k])

        for id,agent_obs in enumerate(multi_agent_obs_list):
            dst = obs_bufs[id].get_obj()
            dst_np = np.frombuffer(dst, dtype=obs_dtype).reshape(obs_shape)  # pylint: disable=W0212
            np.copyto(dst_np, agent_obs)

    def _write_phy(multi_agent_phy_list):
        
        for id,agent_phy in enumerate(multi_agent_phy_list):
            dst = phy_bufs[id].get_obj()
            dst_np = np.frombuffer(dst, dtype=obs_dtype).reshape(phy_shape)  # pylint: disable=W0212
            np.copyto(dst_np, agent_phy)

    env = env_fn_wrapper.x()
    parent_pipe.close()
    try:
        while True:
            cmd, data = pipe.recv()
            if cmd == 'reset':
                multi_informed_time_step:tp.List[InformedTimeStep] = env.reset(data)
                obs = [time_step.observation for time_step in multi_informed_time_step]
                pipe.send(_write_obs(obs))
            elif cmd == 'step':
                #obs, reward, done, info = env.step(data)
                multi_informed_time_step:tp.List[InformedTimeStep] = env.step(data)
                # if all([obs.step_type.last() for obs in informed_time_step]):
                #     done = True
                # else: done = False
                # if done:
                #     obs = env.reset()
                obs = [time_step.observation for time_step in multi_informed_time_step]
                reward = [time_step.action for time_step in multi_informed_time_step]
                discount = [time_step.discount for time_step in multi_informed_time_step]
                info = [time_step.info for time_step in multi_informed_time_step]
                phy = [time_step.physics for time_step in multi_informed_time_step]
                pipe.send((_write_obs(obs), reward, discount, info, _write_phy(phy)))
            elif cmd == 'render':
                pipe.send(env.render(return_rgb=True))
            elif cmd == 'close':
                pipe.send(None)
                break
            elif cmd == 'talk2Env':
                pipe.send(env.talk2Env(data))
            else:
                raise RuntimeError('Got unrecognized cmd %s' % cmd)
    except KeyboardInterrupt:
        print('ShmemVecEnv worker: got KeyboardInterrupt')
    finally:
        env.close()

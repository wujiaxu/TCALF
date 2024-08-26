"""
An interface for asynchronous vectorized environments.
"""

import dis
import multiprocessing as mp
import numpy as np
from robust_navi.robot_crowd_sim.vec_env.vec_env import VecEnv, CloudpickleWrapper, clear_mpi_env_vars
import ctypes
import typing as tp

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

    def __init__(self, env_fns, agent_nums, spaces=None, context='spawn'):
        """
        If you don't specify observation_space, we'll have to create a dummy
        environment to get it.
        """
        ctx = mp.get_context(context)

        if spaces:
            observation_space, action_space = spaces
        else:
            # with logger.scoped_configure(format_strs=[]):
            dummy = env_fns[0]()
            observation_space, action_space = dummy.observation_spec(), dummy.action_spec()
            dummy.close()
            del dummy
        VecEnv.__init__(self, len(env_fns), observation_space, action_space)
        self.agent_nums=agent_nums
        self.obs_keys = list(observation_space.keys()) 
        self.obs_shapes = []
        self.obs_bufs = []
        
        for i in range(len(env_fns)):
            self.obs_bufs.append({})
            self.obs_shapes.append({})
            for k in observation_space.keys():
                if k == 'full_state':
                    obs_shape_k = (8*self.agent_nums[i]+7,)
                else:
                    obs_shape_k = observation_space[k]
                self.obs_bufs[i][k]= ctx.Array(ctypes.c_float,  int(np.prod(obs_shape_k)))
                self.obs_shapes[i][k]=obs_shape_k
        
        self.action_to_pipe = {}
        for i, agent_num in enumerate(self.agent_nums):
            if agent_num not in self.action_to_pipe:
                self.action_to_pipe[agent_num] = []
            self.action_to_pipe[agent_num].append(i)

        self.parent_pipes = []
        self.procs = []
        with clear_mpi_env_vars():
            for env_fn, obs_buf,obs_shape in zip(env_fns, self.obs_bufs,self.obs_shapes):
                wrapped_fn = CloudpickleWrapper(env_fn)
                parent_pipe, child_pipe = ctx.Pipe()
                proc = ctx.Process(target=_subproc_worker,
                            args=(child_pipe, 
                                  parent_pipe, 
                                  wrapped_fn, 
                                  obs_buf, 
                                  self.obs_keys,
                                  obs_shape
                                  ))
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
        """
        actions (dict): 
            robot action (ndarray): env_numX2 
            crowd action (dict): 
                agent_num=1: env(1)X2
                2: env(2)X2
                ...
                N:env(N)X2
        """
        assert len(actions["robot_action"]) == len(self.parent_pipes) #TODO
        crowd_actions_dict = actions["crowd_action"]
        for k in crowd_actions_dict.keys():
            pipe_ids = self.action_to_pipe[k]
            for i, pipe_id in enumerate(pipe_ids):
                self.parent_pipes[pipe_id].send(
                    ('step',
                     {
                         "robot_action":actions["robot_action"][pipe_id],
                         "crowd_action":crowd_actions_dict[k][i]
                      })
                    )
        self.waiting_step = True

    def step_wait(self):
        obs = []
        robot_rews = []
        crowd_rews = []
        robot_discounts = []
        human_discounts = []
        infos = []
        crowd_infos = []
        for pipe in self.parent_pipes:
            o, (rr,cr), (rd,hd), (info,crowd_info) = pipe.recv()
            obs.append(o)
            robot_rews.append(rr)
            crowd_rews.append(cr)
            robot_discounts.append(rd)
            human_discounts.append(hd)
            infos.append(info)
            crowd_infos.append(crowd_info)
        self.waiting_step = False
        crowd_obs, robot_obs = self._decode_obses(obs)
        crowd_rews_dict, human_discounts_dict,crowd_infos_dict = self._decode_by_human_num(crowd_rews,human_discounts,crowd_infos)
        return (
            (crowd_obs, robot_obs), 
            (np.array(robot_rews), crowd_rews_dict), 
            (np.array(robot_discounts),human_discounts_dict), 
            (infos,crowd_infos_dict)
        )
        

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

    def get_images(self):
        for pipe in self.parent_pipes:
            pipe.send(('render', "return_rgb"))
        return [pipe.recv() for pipe in self.parent_pipes]

    def _decode_by_human_num(self, crowd_rews,human_discounts,crowd_infos):

        crowd_rews_dict = {}
        human_discounts_dict = {}
        crowd_infos_dict = {}
        for i,agent_num in enumerate(self.agent_nums):
            if agent_num not in crowd_rews_dict.keys():
                crowd_rews_dict[agent_num] = []
            if agent_num not in human_discounts_dict:
                human_discounts_dict[agent_num] = []
            if agent_num not in crowd_infos_dict:
                crowd_infos_dict[agent_num] = []
            crowd_rews_dict[agent_num].append(crowd_rews[i])
            human_discounts_dict[agent_num].append(human_discounts[i])
            crowd_infos_dict[agent_num].append(crowd_infos[i])

        for k in crowd_rews_dict.keys():
            crowd_rews_dict[k]=np.vstack(crowd_rews_dict[k])
        for k in human_discounts_dict.keys():
            human_discounts_dict[k]=np.vstack(human_discounts_dict[k])
        return crowd_rews_dict,human_discounts_dict,crowd_infos_dict
    
    def _decode_obses(self, all_obs):
        """
        obs : [{full_state:N X full_state_shape,crowd_preference:, scan:...}, 
               {full_state:L X full_state_shape,crowd_preference:, scan:...},
              ...
               {full_state:M X full_state_shape,crowd_preference:, scan:...}]
        """
        crowd_obs, robot_obs = {}, {}
        
        for i,buf in enumerate(self.obs_bufs):
            agent_num = self.agent_nums[i]
            if agent_num not in crowd_obs.keys():
                crowd_obs[agent_num] = {"full_state":[],"crowd_preference":[]}
            for k in self.obs_keys:
                if k == 'full_state' or k=='crowd_preference':
                    crowd_obs[agent_num][k].append(np.frombuffer(
                    buf[k].get_obj(), 
                    dtype=np.float32).reshape(self.obs_shapes[i][k]))
                else:
                    if k not in robot_obs.keys():
                        robot_obs[k] = []
                    robot_obs[k].append(np.frombuffer(
                    buf[k].get_obj(), 
                    dtype=np.float32).reshape(self.obs_shapes[i][k]))
        for agent_num in crowd_obs.keys():
            for k in crowd_obs[agent_num].keys():   
                crowd_obs[agent_num][k]=np.vstack(crowd_obs[agent_num][k])
        for k in robot_obs.keys():
            robot_obs[k]=np.vstack(robot_obs[k])
        return crowd_obs, robot_obs


def _subproc_worker(pipe, parent_pipe, env_fn_wrapper, obs_bufs,obs_keys,obs_shapes):
    """
    Control a single environment instance using IPC and
    shared memory.
    """
    def _write_obs(obs_dict):
        for k in obs_keys:
            dst = obs_bufs[k].get_obj()
            dst_np = np.frombuffer(dst, dtype=np.float32).reshape(obs_shapes[k])  # pylint: disable=W0212
            np.copyto(dst_np, obs_dict[k])

    env = env_fn_wrapper.x()
    parent_pipe.close()
    try:
        while True:
            cmd, data = pipe.recv()
            if cmd == 'reset':
                obs = env.reset(data)
                pipe.send(_write_obs(obs))
            elif cmd == 'step':
                obs,  (robot_reward, crowd_reward), (robot_discount,human_discount), (info,crowd_info) = env.step(data)
                if robot_discount==0.0 and human_discount==0.0:
                    done = True
                else: done = False
                if done:
                    obs = env.reset()
                pipe.send((_write_obs(obs),  (robot_reward, crowd_reward), (robot_discount,human_discount), (info,crowd_info)))

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

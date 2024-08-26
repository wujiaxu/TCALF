"""
Helpers for dealing with vectorized environments.
"""

from collections import OrderedDict

# import gym
import numpy as np
from controllable_navi.crowd_sim.crowd_sim import NaviObsSpace

def copy_obs_dict(obs):
    """
    Deep-copy an observation dict.
    """
    return {k: np.copy(v) for k, v in obs.items()}


def dict_to_obs(obs_dict):
    """
    Convert an observation dict into a raw array if the
    original observation space was not a Dict space.
    """
    if set(obs_dict.keys()) == {None}:
        return obs_dict[None]
    return obs_dict


"""
NaviObsSpace({"scan":(720,),"robot_state":(7,)}, dtype=np.float32, name='relative pose to goal')

class NaviObsSpace:

  def __init__(self, shapes:dict, dtype, name: str):
    self._shape_total = 0
    self._shapes = {}
    for compo_name in shapes.keys():
        dim = 1
        shape = shapes[compo_name]
        for d in shape:
            dim*=d
        self._shape_total+=dim
        self._shapes[compo_name] = (shape,dim)
    self._dtype = np.dtype(dtype)
    self._name = name

  def get_shape(self,name):
      return self._shapes[name]

  @property
  def shape_dict(self):
    return self._shapes
  @property
  def shape(self):
    return (self._shape_total,)

  @property
  def dtype(self):
    return self._dtype

  @property
  def name(self):
    return self._name    
"""
def obs_space_info(obs_space:NaviObsSpace):
    """
    Get dict-structured information about a gym.Space.

    Returns:
      A tuple (keys, shapes, dtypes):
        keys: a list of dict keys.
        shapes: a dict mapping keys to shapes.
        dtypes: a dict mapping keys to dtypes.
    """
    # if isinstance(obs_space, gym.spaces.Dict): #TODO
    #     assert isinstance(obs_space.spaces, OrderedDict)
    #     subspaces = obs_space.spaces
    # elif isinstance(obs_space, gym.spaces.Tuple): #TODO
    #     assert isinstance(obs_space.spaces, tuple)
    #     subspaces = {i: obs_space.spaces[i] for i in range(len(obs_space.spaces))}
    # else:
    #     subspaces = {None: obs_space}
    keys = []
    shapes = {}
    dtypes = {}
    for key, (shape,dim) in obs_space.shape_dict:
        keys.append(key)
        shapes[key] = shape
        dtypes[key] = obs_space.dtype
    return keys, shapes, dtypes


def obs_to_dict(obs):
    """
    Convert an observation into a dict.
    """
    if isinstance(obs, dict):
        return obs
    return {None: obs}

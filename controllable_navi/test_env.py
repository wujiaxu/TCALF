from pathlib import Path
import sys
import faulthandler
faulthandler.enable()

base = Path(__file__).absolute().parents[1]
# we need to add base repo to be able to import controllable_navi
# we need to add controllable_navi to be able to reload legacy checkpoints
for fp in [base, base / "controllable_navi"]:
    assert fp.exists()
    if str(fp) not in sys.path:
        sys.path.append(str(fp))
from controllable_navi.crowd_sim.multi_robot_sim import MultiRobotSimConfig,build_multirobotworld_task
import numpy as np
from controllable_navi.vec_env.envs import make_vec_envs

cfg = MultiRobotSimConfig()
# cfg.max_robot_num = 3
# crowd_sim = build_multirobotworld_task(cfg,"PointGoalNavi","train")
# print(crowd_sim.observation_spec())
# step_ex = crowd_sim.reset()
# # crowd_sim.robots[0].set(0.3,0,1,1,0,0,0)
# # crowd_sim.robots[1].set(0,0,1,1,0,0,0)
# step_ex = crowd_sim.step(np.array([[1,0],[1,0],[1,0]]))
# print(crowd_sim._step_info[0].info_list)
# print(np.min(step_ex[0].observation[:-7]))
# print()
# print(crowd_sim._step_info[1].info_list)
# print(np.min(step_ex[1].observation[:-7]))

def get_random_human_num(max_human_num,nenv):
    return np.random.randint(2,max_human_num+1,nenv)
nenv=32
human_nums = get_random_human_num(5,nenv)
print(sum(human_nums))
envs = make_vec_envs(cfg,1,nenv,0.9,'PointGoalNavi',human_nums,0.99,"time_aware_raw_scan",50,"cpu",wrap_pytorch=False)
cfg.max_robot_num = 5
actions = [np.ones((human_num,2))*np.random.random() for human_num in human_nums]
obs = envs.reset()
obs,reward,discount,infos,phy = envs.step(actions)
print(obs.shape)
obs,reward,discount,infos,phy = envs.step(actions)
print(obs)
envs.close()

# envs = make_vec_envs(cfg,1,1,0.9,'PointGoalNavi',[3],0.99,"time_aware_raw_scan",50,"cpu",wrap_pytorch=False)
# cfg.max_robot_num = 5
# action1 = np.ones((3,2))
# obs = envs.reset()
# obs,reward,discount,infos,phy = envs.step([action1])
# print(obs.shape,np.max(obs[:,:-7],axis=1),obs[:,-7:])
# obs,reward,discount,infos,phy = envs.step([action1])
# print(obs[:,-7:])
# envs.close()
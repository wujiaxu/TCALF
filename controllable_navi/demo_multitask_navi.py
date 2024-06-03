# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""
Useful links:
Streamlit cheatsheet:
https://docs.streamlit.io/library/cheatsheet

Also check the components we provide for demos in metastreamlit:
https://github.com/fairinternal/metastreamlit
You can request new components by creating an issue
"""

# Designed to run from controllable_agent with streamlit run demo/main.py
# import os
# os.environ["CUDA_VISIBLE_DEVICES"] = ""  # avoid using CUDA
import sys
import time
import logging
import tempfile
from pathlib import Path
from collections import OrderedDict
from omegaconf import OmegaConf
import hydra
from hydra.core.config_store import ConfigStore
import dataclasses
import typing as tp
import omegaconf as omgcf
# import streamlit as st
try:
    import controllable_navi
    base = Path(controllable_navi.__file__).absolute().parents[1]
except ImportError:
    base = Path(__file__).absolute().parents[1]
# we need to add base repo to be able to import controllable_navi
# we need to add controllable_navi to be able to reload legacy checkpoints
for fp in [base,base / "controllable_navi"]:
    assert fp.exists()
    if str(fp) not in sys.path:
        sys.path.append(str(fp))
print("base", base)
from controllable_navi import pretrain
import numpy as np
import torch
import torch.nn.functional as F
from controllable_navi import runner
from controllable_navi import goals
from controllable_navi import utils
from controllable_navi.video import VideoRecorder
logger = logging.getLogger(__name__)

import matplotlib.pyplot as plt

@dataclasses.dataclass
class TestConfig():
    model_dir:str = "2024.05.26/062410_gd_aps_crowdnavi_PointGoalNavi_online"
    num_eval_episodes:int=10
    task:str = 'PassLeftSide'

ConfigStore.instance().store(name="test_workspace_config", node=TestConfig)

def load_workspace(model_dir: str):
    model_base=Path('/home/dl/wu_ws/TCALF/controllable_navi/exp_local/')
    checkpoint = model_base/Path(model_dir)/"models/latest.pt"
    config_file = model_base/Path(model_dir)/".hydra/config.yaml"
    hp = runner.HydraEntryPoint(base / "controllable_navi/pretrain.py")
    cfg = OmegaConf.load(config_file)
    cfg.use_tb=0
    cfg.use_hiplog=0
    ws = hp.workspace(cfg)
    # ws.train_env.base_env.init_render_ax(ax)
    # ws.train_env.reset()
    with checkpoint.open("rb") as f:
        payload = torch.load(f, map_location=ws.device)
    ws.agent = payload["agent"]
    ws.agent.cfg.device = ws.cfg.device
    replay = payload["replay_loader"]
    ws.replay_loader = replay
    ws.replay_storage = replay
    return ws

@hydra.main(config_path='.', config_name='test_config', version_base="1.1")
def main(test_cfg: omgcf.DictConfig) -> None:

    # load
    ws = load_workspace(test_cfg.model_dir)
    ws.finalize(num_eval_episodes=test_cfg.num_eval_episodes,custom_task=test_cfg.task)


if __name__ == '__main__':
    main()

    
    # recorder = VideoRecorder(base, camera_id=ws.video_recorder.camera_id, use_wandb=False)
    # recorder.enabled = True


# # reward = goals.WalkerEquation("x")
# # reward._precompute_for_demo(ws)  # precompute before first run
# # ws.replay_loader._storage.clear()  # clear memory since not used anymore
# # params = list(reward._extract(reward._env))
# # params_str = ", ".join(f"`{x}`" for x in params)

# # st.write("##### Try Your Own Reward Function for Walker")
# # st.write(f"Enter a crowdnavi reward function to maximize")
# # string = st.text_input("Reward function:", value=st.session_state.get("prefill", ""))
# # st.session_state.pop("prefill", None)
# reward_texts = [
#     ("PointGoalNavi", "PointGoalNavi"),
#     ("Tracking", "Tracking"),
#     ("PassLeftSide", "PassLeftSide"),
#     ("PassRightSide", "PassRightSide"),
#     ("FollowWall", "FollowWall"),
#     ("AwayFromHuman", "AwayFromHuman"),
#     ("LowSpeed","LowSpeed"),
# ]
# string = "PointGoalNavi"

# if string and string is not None:
#     reward = goals.CrowdNaviReward(string)
#     logger.info(f"Running reward: {string}")  # for the console
#     start = time.time()
#     meta = pretrain._init_eval_meta(ws, custom_reward=reward)
#     end = time.time()

#     print("infer meta used: {} s".format(end-start))
#     # play
#     env = ws._make_env("test")
#     time_step = env.reset()
#     # recorder.init(env)
#     total_reward = 0
#     durations = dict(model=0.0, env=0.0, render=0.0)
#     t_start = time.time()
#     while time_step.last():
#         t0 = time.time()
#         with torch.no_grad(), utils.eval_mode(ws.agent):
#             action = ws.agent.act(time_step.observation,
#                                   meta,
#                                   1000000,
#                                   eval_mode=True)
#         t1 = time.time()
#         time_step = env.step(action)
        
#         t2 = time.time()
#         # recorder.record(env)
#         # env.render(return_rgb=False)
#         t3 = time.time()
#         durations["model"] += t1 - t0
#         durations["env"] += t2 - t1
#         durations["render"] += t3 - t2
#         total_reward += time_step.reward #reward.from_env(env)
        
#     print(f"Total play time {time.time() - t_start:.2f}s with {durations}")
#     print(f"Reward is {total_reward}\n\n")
    # state = reward._extract(env)
    # state_str = " ".join(f"{x}={y:.2f}" for x, y in state.items())
    # name = string+"_demo.mp4"
    # with tempfile.TemporaryDirectory() as tmp:
    #     recorder.save_dir = Path(tmp)
    #     t0 = time.time()
    #     recorder.save(name)
    #     print(f"Saved video to {recorder.save_dir / name} in {time.time() - t0:.2f}s, now serving it.")






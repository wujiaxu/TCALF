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
# import streamlit as st
try:
    import url_benchmark
    base = Path(url_benchmark.__file__).absolute().parents[1]
except ImportError:
    base = Path(__file__).absolute().parents[1]
# we need to add base repo to be able to import url_benchmark
# we need to add url_benchmar to be able to reload legacy checkpoints
for fp in [base, base / "url_benchmark"]:
    assert fp.exists()
    if str(fp) not in sys.path:
        sys.path.append(str(fp))
print("base", base)
from url_benchmark import pretrain
import numpy as np
import torch
import torch.nn.functional as F
from controllable_agent import runner
from url_benchmark import goals
from url_benchmark import utils
from url_benchmark.video import VideoRecorder
logger = logging.getLogger(__name__)

# st.set_page_config(
#     page_title="Controllable agent - Meta AI",
#     menu_items={"About": "This demo is powered by the code available at https://github.com/facebookresearch/controllable_agent\nCopyright 2022 Meta Inc. Available under MIT Licence."},

# )
# # st.title('Controllable agent')
# st.sidebar.write('# Controllable Agent Demo')
# st.sidebar.write("### Optimize Any Reward Function with a Single Pretrained Agent")
# st.sidebar.write("***Ahmed Touati, Jérémy Rapin, Yann Ollivier***")
# st.sidebar.write("A controllable agent is a reinforcement learning agent whose _reward function can be set in real time_, without any additional learning or fine-tuning, based on a reward-free pretraining phase.")
# st.sidebar.write("""The controllable agent here uses the _forward-backward representation_ from our papers:
# * [Does Zero-Shot Reinforcement Learning Exist?](https://arxiv.org/abs/2209.14935)
# * [Learning One Representation to Optimize All Rewards](https://arxiv.org/abs/2103.07945) (Neurips 2021)
# """)
# st.sidebar.write("The [code is open-source](https://github.com/facebookresearch/controllable_agent).")

model_path = Path("/home/dl/wu_ws/TCALF/url_benchmark/exp_local/2024.04.09/234316_aps_crowdnavi_PointGoalNavi_online/models")
# if not model_path.exists():
#     model_path = base / "models"

# having more cases will trigger a dropdown box
CASES = {
    # Update the following path to a checkpoint that exists in you system
    "crowdnavi - 240409 (rnd init)": model_path / "latest.pt",
}
CASES = {x: y for x, y in CASES.items() if y.exists()}
# if len(CASES) > 1:
#     case = st.selectbox(
#         'Which model do you want to load?',
#         list(CASES)
#     )
# else:
case = list(CASES)[0]
assert case is not None


# @st.cache(max_entries=1, allow_output_mutation=True)
def load_workspace(case: str):
    checkpoint = CASES[case]
    hp = runner.HydraEntryPoint(base / "url_benchmark/anytrain.py")
    ws = hp.workspace(task="crowdnavi_PointGoalNavi", replay_buffer_episodes=100)
    ws.train_env.reset()
    with checkpoint.open("rb") as f:
        payload = torch.load(f, map_location=ws.device)
    ws.agent = payload["agent"]
    ws.agent.cfg.device = ws.cfg.device
    replay = payload["replay_loader"]
    ws.replay_loader = replay
    ws.replay_storage = replay
    return ws


# load
ws = load_workspace(case)
recorder = VideoRecorder(base, camera_id=ws.video_recorder.camera_id, use_wandb=False)
recorder.enabled = True


# reward = goals.WalkerEquation("x")
# reward._precompute_for_demo(ws)  # precompute before first run
# ws.replay_loader._storage.clear()  # clear memory since not used anymore
# params = list(reward._extract(reward._env))
# params_str = ", ".join(f"`{x}`" for x in params)

# st.write("##### Try Your Own Reward Function for Walker")
# st.write(f"Enter a crowdnavi reward function to maximize")
# string = st.text_input("Reward function:", value=st.session_state.get("prefill", ""))
# st.session_state.pop("prefill", None)
reward_texts = [
    ("PointGoalNavi", "PointGoalNavi"),
    ("Tracking", "Tracking"),
    ("PassLeftSide", "PassLeftSide"),
    ("PassRightSide", "PassRightSide"),
    ("FollowWall", "FollowWall"),
    ("AwayFromHuman", "AwayFromHuman"),
    ("LowSpeed","LowSpeed"),
]
string = "PointGoalNavi"

if string and string is not None:
    reward = goals.CrowdNaviReward(string)
    logger.info(f"Running reward: {string}")  # for the console
    start = time.time()
    meta = pretrain._init_eval_meta(ws, custom_reward=reward)
    end = time.time()

    print("infer meta used: {} s".format(end-start))
    # play
    env = ws._make_env()
    time_step = env.reset()
    # recorder.init(env)
    total_reward = 0
    k = 0
    durations = dict(model=0.0, env=0.0, render=0.0)
    t_start = time.time()
    while k < 200 and not time_step.last():
        k += 1
        t0 = time.time()
        with torch.no_grad(), utils.eval_mode(ws.agent):
            action = ws.agent.act(time_step.observation,
                                  meta,
                                  5000,
                                  eval_mode=True)
        t1 = time.time()
        time_step = env.step(action)
        
        t2 = time.time()
        # recorder.record(env)
        env.render(return_rgb=False)
        t3 = time.time()
        durations["model"] += t1 - t0
        durations["env"] += t2 - t1
        durations["render"] += t3 - t2
        total_reward += time_step.reward #reward.from_env(env)
        
    print(f"Total play time {time.time() - t_start:.2f}s with {durations}")
    print(f"Average reward is {total_reward / k}\n\n")
    # state = reward._extract(env)
    # state_str = " ".join(f"{x}={y:.2f}" for x, y in state.items())
    # name = string+"_demo.mp4"
    # with tempfile.TemporaryDirectory() as tmp:
    #     recorder.save_dir = Path(tmp)
    #     t0 = time.time()
    #     recorder.save(name)
    #     print(f"Saved video to {recorder.save_dir / name} in {time.time() - t0:.2f}s, now serving it.")






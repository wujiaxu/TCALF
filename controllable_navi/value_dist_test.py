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
    model_dir:str = "2024.06.25/091918_ddpg_crowdnavi_PointGoalNavi_online"
    task:str = 'value_dist'

ConfigStore.instance().store(name="test_workspace_config", node=TestConfig)

def load_workspace(model_dir: str):
    model_base=Path('/home/dl/wu_ws/TCALF/controllable_navi/exp_local/')
    checkpoint = model_base/Path(model_dir)/"models/latest.pt"
    config_file = model_base/Path(model_dir)/".hydra/config.yaml"
    hp = runner.HydraEntryPoint(base / "controllable_navi/pretrain_self_play.py")
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
    ws.agent.train(False)
    ws.eval_env = ws._make_env(phase='test')
    for case in range(-8,0):
        time_step_multi = ws.eval_env.reset(test_case=case)
        with torch.no_grad(), utils.eval_mode(ws.agent):
            for time_step in time_step_multi:
                value,action = ws.agent.get_value(time_step.observation,OrderedDict(),30000)
                print(value,action)

if __name__ == '__main__':
    main()

    
    
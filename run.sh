#!/bin/sh
#PJM -L rscgrp=debug-a
#PJM -L node=1
#PJM -L elapse=0:15:00
#PJM -L jobenv=singularity
#PJM -g group1
#PJM -j
module load singularity/3.7.3
singularity build controllable_navi.sif controllable_navi.def
singularity run --nv controllable_navi.sif python /opt/TCALF/controllable_navi/pretrain.py device=cuda:2 agent=gd_aps agent.use_self_supervised_encoder=False agent.sf_dim=5 agent.use_sequence=False agent.balancing_factor=1.0 agent.lagrange_multiplier_upper_bound=10. agent.lagrangian_k_p=1 agent.lagrangian_k_i=0.0003 agent.lagrangian_k_d=0 task=crowdnavi_PointGoalNavi use_tb=1 use_hiplog=1 replay_buffer_episodes=6000 discount=0.99 obs_type=time_aware_raw_scan agent.lr=0.000001 agent.critic_target_tau=0.001 reward_free=True num_seed_frames=2000 save_video=True crowd_sim=CrowdWorld crowd_sim.max_human_num=1 crowd_sim.with_static_obstacle=False crowd_sim.map_size=6 crowd_sim.reward_goal=0.25 crowd_sim.goal_factor=2 crowd_sim.penalty_collision=-0.25 agent.sequence_encoder_config.transformer.num_layers=1
#!/bin/bash

conda_env=url_navi
# Define an array of session names, conda environments, and corresponding Python scripts
declare -a sessions_and_scripts=(
    # "ddpg_3_wo:pretrain.py device=cuda:0 agent=ddpg task=crowdnavi_PointGoalNavi use_tb=1 use_hiplog=1 replay_buffer_episodes=2000 discount=0.99 obs_type=raw_scan agent.lr=0.000001 agent.critic_target_tau=0.001 reward_free=False num_seed_frames=500 save_video=True crowd_sim=CrowdWorld crowd_sim.max_human_num=3 crowd_sim.with_static_obstacle=False crowd_sim.map_size=6 crowd_sim.reward_goal=0.25 crowd_sim.goal_factor=10 crowd_sim.penalty_collision=-0.25"
    # "ddpg_3_w:pretrain.py device=cuda:1 agent=ddpg task=crowdnavi_PointGoalNavi use_tb=1 use_hiplog=1 replay_buffer_episodes=2000 discount=0.99 obs_type=raw_scan agent.lr=0.000001 agent.critic_target_tau=0.001 reward_free=False num_seed_frames=500 save_video=True crowd_sim=CrowdWorld crowd_sim.max_human_num=3 crowd_sim.with_static_obstacle=True crowd_sim.map_size=6 crowd_sim.reward_goal=0.25 crowd_sim.goal_factor=10 crowd_sim.penalty_collision=-0.25"
    # "ddpg_3_wo_tau0001:pretrain.py agent=ddpg task=crowdnavi_PointGoalNavi use_tb=1 use_hiplog=1 replay_buffer_episodes=2000 discount=0.99 obs_type=raw_scan agent.lr=0.000001 agent.critic_target_tau=0.001 reward_free=False num_seed_frames=500 save_video=True crowd_sim=CrowdWorld crowd_sim.max_human_num=3 crowd_sim.with_static_obstacle=False crowd_sim.map_size=6"
    # "crowd_aps_3_wo_alpha_09_glr_0003:pretrain.py device=cuda:0 agent=crowd_aps agent.constrain_relaxation_alpha=0.9 agent.lagrangian_multiplier_lr=0.003 task=crowdnavi_PointGoalNavi use_tb=1 use_hiplog=1 replay_buffer_episodes=2000 discount=0.99 obs_type=raw_scan agent.lr=0.000001 agent.critic_target_tau=0.001 reward_free=True num_seed_frames=4000 save_video=True crowd_sim=CrowdWorld crowd_sim.max_human_num=3 crowd_sim.with_static_obstacle=False crowd_sim.map_size=6 crowd_sim.reward_goal=0.25 crowd_sim.goal_factor=0.2 crowd_sim.penalty_collision=-0.25"
    # "crowd_aps_3_wo_alpha_09_glr_00003:pretrain.py device=cuda:1 agent=crowd_aps agent.constrain_relaxation_alpha=0.9 agent.lagrangian_multiplier_lr=0.0003 task=crowdnavi_PointGoalNavi use_tb=1 use_hiplog=1 replay_buffer_episodes=2000 discount=0.99 obs_type=raw_scan agent.lr=0.000001 agent.critic_target_tau=0.001 reward_free=True num_seed_frames=4000 save_video=True crowd_sim=CrowdWorld crowd_sim.max_human_num=3 crowd_sim.with_static_obstacle=False crowd_sim.map_size=6 crowd_sim.reward_goal=0.25 crowd_sim.goal_factor=0.2 crowd_sim.penalty_collision=-0.25"
    # "crowd_aps_3_wo_alpha_09_glr_000003:pretrain.py device=cuda:2 agent=crowd_aps agent.constrain_relaxation_alpha=0.9 agent.lagrangian_multiplier_lr=0.00003 task=crowdnavi_PointGoalNavi use_tb=1 use_hiplog=1 replay_buffer_episodes=2000 discount=0.99 obs_type=raw_scan agent.lr=0.000001 agent.critic_target_tau=0.001 reward_free=True num_seed_frames=4000 save_video=True crowd_sim=CrowdWorld crowd_sim.max_human_num=3 crowd_sim.with_static_obstacle=False crowd_sim.map_size=6 crowd_sim.reward_goal=0.25 crowd_sim.goal_factor=0.2 crowd_sim.penalty_collision=-0.25"
    # "gd_aps_3_wo_sf_50_alpha_0025:pretrain.py device=cuda:0 agent=gd_aps agent.sf_dim=50 agent.balancing_factor=0.025 task=crowdnavi_PointGoalNavi use_tb=1 use_hiplog=1 replay_buffer_episodes=2000 discount=0.99 obs_type=raw_scan agent.lr=0.000001 agent.critic_target_tau=0.001 reward_free=True num_seed_frames=2000 save_video=True crowd_sim=CrowdWorld crowd_sim.max_human_num=3 crowd_sim.with_static_obstacle=False crowd_sim.map_size=6 crowd_sim.reward_goal=0.25 crowd_sim.goal_factor=0.2 crowd_sim.penalty_collision=-0.25"
    # "gd_aps_3_wo_sf_50_alpha_005:pretrain.py device=cuda:1 agent=gd_aps agent.sf_dim=50 agent.balancing_factor=0.05 task=crowdnavi_PointGoalNavi use_tb=1 use_hiplog=1 replay_buffer_episodes=2000 discount=0.99 obs_type=raw_scan agent.lr=0.000001 agent.critic_target_tau=0.001 reward_free=True num_seed_frames=2000 save_video=True crowd_sim=CrowdWorld crowd_sim.max_human_num=3 crowd_sim.with_static_obstacle=False crowd_sim.map_size=6 crowd_sim.reward_goal=0.25 crowd_sim.goal_factor=0.2 crowd_sim.penalty_collision=-0.25"
    # "gd_aps_3_wo_sf_50_alpha_0075:pretrain.py device=cuda:2 agent=gd_aps agent.sf_dim=50 agent.balancing_factor=0.075 task=crowdnavi_PointGoalNavi use_tb=1 use_hiplog=1 replay_buffer_episodes=2000 discount=0.99 obs_type=raw_scan agent.lr=0.000001 agent.critic_target_tau=0.001 reward_free=True num_seed_frames=2000 save_video=True crowd_sim=CrowdWorld crowd_sim.max_human_num=3 crowd_sim.with_static_obstacle=False crowd_sim.map_size=6 crowd_sim.reward_goal=0.25 crowd_sim.goal_factor=0.2 crowd_sim.penalty_collision=-0.25"
    # "gd_aps_toy:pretrain.py device=cuda:2 agent=gd_aps agent.use_self_supervised_encoder=False agent.sf_dim=5 agent.use_sequence=True agent.balancing_factor=1.0 agent.lagrange_multiplier_upper_bound=10. agent.sf_reward_target=25 agent.lagrangian_k_p=1 agent.lagrangian_k_i=0.0003 agent.lagrangian_k_d=0 task=crowdnavi_PointGoalNavi use_tb=1 use_hiplog=1 replay_buffer_episodes=6000 discount=0.99 obs_type=time_aware_toy_state agent.lr=0.000001 agent.critic_target_tau=0.001 reward_free=True num_seed_frames=2000 save_video=True crowd_sim=CrowdWorld crowd_sim.max_human_num=1 crowd_sim.with_static_obstacle=False crowd_sim.map_size=6 crowd_sim.reward_goal=0.25 crowd_sim.goal_factor=2 crowd_sim.penalty_collision=-0.25"
    # "gd_aps_stoy:pretrain.py device=cuda:1 agent=gd_aps agent.use_self_supervised_encoder=False agent.sf_dim=5 agent.use_sequence=True agent.balancing_factor=1.0 agent.lagrange_multiplier_upper_bound=10. agent.sf_reward_target=25 agent.lagrangian_k_p=1 agent.lagrangian_k_i=0.0003 agent.lagrangian_k_d=0 task=crowdnavi_PointGoalNavi use_tb=1 use_hiplog=1 replay_buffer_episodes=6000 discount=0.99 obs_type=time_aware_toy_state agent.lr=0.000001 agent.critic_target_tau=0.001 reward_free=True num_seed_frames=2000 save_video=True crowd_sim=CrowdWorld crowd_sim.max_human_num=1 crowd_sim.with_static_obstacle=False crowd_sim.map_size=6 crowd_sim.reward_goal=10 crowd_sim.goal_factor=2 crowd_sim.penalty_collision=-10"
    "gd_aps_scn_1:pretrain.py device=cuda:1 agent=gd_aps agent.use_sequence=False agent.max_constraint_value=1.3 agent.lagrangian_k_p=1 agent.lagrangian_k_i=0.0003 agent.lagrangian_k_d=0 task=crowdnavi_PointGoalNavi use_tb=1 use_hiplog=1 replay_buffer_episodes=6000 discount=0.99 obs_type=time_aware_raw_scan agent.lr=0.000001 agent.critic_target_tau=0.001 reward_free=True num_seed_frames=2000 save_video=True crowd_sim=CrowdWorld crowd_sim.max_human_num=1 crowd_sim.with_static_obstacle=False crowd_sim.map_size=6 crowd_sim.reward_goal=0.25 crowd_sim.goal_factor=2 crowd_sim.penalty_collision=-0.25 agent.sequence_encoder_config.transformer.num_layers=1"
    "gd_aps_scn_2:pretrain.py device=cuda:2 agent=gd_aps agent.use_sequence=False agent.max_constraint_value=1.5 agent.lagrangian_k_p=1 agent.lagrangian_k_i=0.0003 agent.lagrangian_k_d=0 task=crowdnavi_PointGoalNavi use_tb=1 use_hiplog=1 replay_buffer_episodes=6000 discount=0.99 obs_type=time_aware_raw_scan agent.lr=0.000001 agent.critic_target_tau=0.001 reward_free=True num_seed_frames=2000 save_video=True crowd_sim=CrowdWorld crowd_sim.max_human_num=1 crowd_sim.with_static_obstacle=False crowd_sim.map_size=6 crowd_sim.reward_goal=0.25 crowd_sim.goal_factor=2 crowd_sim.penalty_collision=-0.25 agent.sequence_encoder_config.transformer.num_layers=1"
    # "gd_aps_toy_EL_50_wo_sf_5_c_12:pretrain.py device=cuda:1 agent=gd_aps agent.use_self_supervised_encoder=False agent.sf_dim=5 agent.balancing_factor=1.0 agent.lagrange_multiplier_upper_bound=10. agent.sf_reward_target=25 agent.lagrangian_k_p=0.5 agent.lagrangian_k_i=0.0003 agent.lagrangian_k_d=0 task=crowdnavi_PointGoalNavi use_tb=1 use_hiplog=1 replay_buffer_episodes=6000 discount=0.99 obs_type=toy_state agent.lr=0.000001 agent.critic_target_tau=0.001 reward_free=True num_seed_frames=2000 save_video=True crowd_sim=CrowdWorld crowd_sim.max_human_num=1 crowd_sim.with_static_obstacle=False crowd_sim.map_size=6 crowd_sim.reward_goal=0.25 crowd_sim.goal_factor=2 crowd_sim.penalty_collision=-0.25"
    # "gd_aps_toy_EL_50_wo_sf_5_c_14:pretrain.py device=cuda:2 agent=gd_aps agent.use_self_supervised_encoder=False agent.sf_dim=5 agent.balancing_factor=1.0 agent.lagrange_multiplier_upper_bound=10. agent.sf_reward_target=25 agent.lagrangian_k_p=0.1 agent.lagrangian_k_i=0.0003 agent.lagrangian_k_d=0 task=crowdnavi_PointGoalNavi use_tb=1 use_hiplog=1 replay_buffer_episodes=6000 discount=0.99 obs_type=toy_state agent.lr=0.000001 agent.critic_target_tau=0.001 reward_free=True num_seed_frames=2000 save_video=True crowd_sim=CrowdWorld crowd_sim.max_human_num=1 crowd_sim.with_static_obstacle=False crowd_sim.map_size=6 crowd_sim.reward_goal=0.25 crowd_sim.goal_factor=2 crowd_sim.penalty_collision=-0.25"
    
    # Add more session, environment, and script pairs as needed
)

# Iterate through each session, environment, and script pair
for item in "${sessions_and_scripts[@]}"
do
    # Split the session, environment, and script information
    IFS=':' read -r session_name script_path <<< "$item"
    
    # Start a new Screen session with the specified name
    # Activate the conda environment and execute the Python script within the session
    screen -dmS "$session_name" bash -c "source activate $conda_env && python $script_path; exec bash"
    
    # Optional: Provide some feedback about the session creation
    echo "Started Screen session '$session_name' running script '$script_path' in conda environment '$conda_env'"

    sleep 5
done

# End of the script
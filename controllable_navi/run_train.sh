#!/bin/bash

conda_env=url_navi
# Define an array of session names, conda environments, and corresponding Python scripts
declare -a sessions_and_scripts=(
    "ddpg_3_wo:pretrain.py device=cuda:0 agent=ddpg task=crowdnavi_PointGoalNavi use_tb=1 use_hiplog=1 replay_buffer_episodes=2000 discount=0.99 obs_type=raw_scan agent.lr=0.000001 agent.critic_target_tau=0.001 reward_free=False num_seed_frames=500 save_video=True crowd_sim=CrowdWorld crowd_sim.max_human_num=3 crowd_sim.with_static_obstacle=False crowd_sim.map_size=6 crowd_sim.reward_goal=0.25 crowd_sim.goal_factor=10 crowd_sim.penalty_collision=-0.25"
    "ddpg_3_w:pretrain.py device=cuda:1 agent=ddpg task=crowdnavi_PointGoalNavi use_tb=1 use_hiplog=1 replay_buffer_episodes=2000 discount=0.99 obs_type=raw_scan agent.lr=0.000001 agent.critic_target_tau=0.001 reward_free=False num_seed_frames=500 save_video=True crowd_sim=CrowdWorld crowd_sim.max_human_num=3 crowd_sim.with_static_obstacle=True crowd_sim.map_size=6 crowd_sim.reward_goal=0.25 crowd_sim.goal_factor=10 crowd_sim.penalty_collision=-0.25"
    # "ddpg_3_wo_tau0001:pretrain.py agent=ddpg task=crowdnavi_PointGoalNavi use_tb=1 use_hiplog=1 replay_buffer_episodes=2000 discount=0.99 obs_type=raw_scan agent.lr=0.000001 agent.critic_target_tau=0.001 reward_free=False num_seed_frames=500 save_video=True crowd_sim=CrowdWorld crowd_sim.max_human_num=3 crowd_sim.with_static_obstacle=False crowd_sim.map_size=6"
    # "crowd_aps_3_wo_alpha_09_glr_0003:pretrain.py device=cuda:0 agent=crowd_aps agent.constrain_relaxation_alpha=0.9 agent.lagrangian_multiplier_lr=0.003 task=crowdnavi_PointGoalNavi use_tb=1 use_hiplog=1 replay_buffer_episodes=2000 discount=0.99 obs_type=raw_scan agent.lr=0.000001 agent.critic_target_tau=0.001 reward_free=True num_seed_frames=4000 save_video=True crowd_sim=CrowdWorld crowd_sim.max_human_num=3 crowd_sim.with_static_obstacle=False crowd_sim.map_size=6 crowd_sim.reward_goal=0.25 crowd_sim.goal_factor=0.2 crowd_sim.penalty_collision=-0.25"
    # "crowd_aps_3_wo_alpha_09_glr_00003:pretrain.py device=cuda:1 agent=crowd_aps agent.constrain_relaxation_alpha=0.9 agent.lagrangian_multiplier_lr=0.0003 task=crowdnavi_PointGoalNavi use_tb=1 use_hiplog=1 replay_buffer_episodes=2000 discount=0.99 obs_type=raw_scan agent.lr=0.000001 agent.critic_target_tau=0.001 reward_free=True num_seed_frames=4000 save_video=True crowd_sim=CrowdWorld crowd_sim.max_human_num=3 crowd_sim.with_static_obstacle=False crowd_sim.map_size=6 crowd_sim.reward_goal=0.25 crowd_sim.goal_factor=0.2 crowd_sim.penalty_collision=-0.25"
    # "crowd_aps_3_wo_alpha_09_glr_000003:pretrain.py device=cuda:2 agent=crowd_aps agent.constrain_relaxation_alpha=0.9 agent.lagrangian_multiplier_lr=0.00003 task=crowdnavi_PointGoalNavi use_tb=1 use_hiplog=1 replay_buffer_episodes=2000 discount=0.99 obs_type=raw_scan agent.lr=0.000001 agent.critic_target_tau=0.001 reward_free=True num_seed_frames=4000 save_video=True crowd_sim=CrowdWorld crowd_sim.max_human_num=3 crowd_sim.with_static_obstacle=False crowd_sim.map_size=6 crowd_sim.reward_goal=0.25 crowd_sim.goal_factor=0.2 crowd_sim.penalty_collision=-0.25"
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
# Decoupled Loco-Manipulation with Force Estimator

## TODO
- [x] Release code backbone
- [x] Release locomotion training pipeline
- [ ] Release motion retargeting pipeline
- [ ] Release sim2sim in MuJoCo
- [ ] Release sim2real with UnitreeSDK


# Installation

## IsaacGym Conda Env

Create mamba/conda environment, in the following we use conda for example, but you can use mamba as well.

```bash
conda create -n hvgym python=3.8
conda activate hvgym
```
### Install IsaacGym

Download [IsaacGym](https://developer.nvidia.com/isaac-gym/download) and extract:

```bash
wget https://developer.nvidia.com/isaac-gym-preview-4
tar -xvzf isaac-gym-preview-4
```

Install IsaacGym Python API:

```bash
pip install -e isaacgym/python
```

Test installation:

```bash
python 1080_balls_of_solitude.py  # or
python joint_monkey.py
```

For libpython error:

- Check conda path:
    ```bash
    conda info -e
    ```
- Set LD_LIBRARY_PATH:
    ```bash
    export LD_LIBRARY_PATH=</path/to/conda/envs/your_env/lib>:$LD_LIBRARY_PATH
    ```

### Install HumanoidVerse

Install dependencies:
```bash
pip install -e .
pip install -e isaac_utils
conda install pinocchio -c conda-forge
```

Test with:
```bash
HYDRA_FULL_ERROR=1 python humanoidverse/train_agent.py \
+simulator=isaacgym \
+exp=locomotion \
+domain_rand=NO_domain_rand \
+rewards=loco/reward_g1_locomotion \
+robot=g1/g1_29dof_anneal_23dof \
+terrain=terrain_locomotion_plane \
+obs=loco/leggedloco_obs_singlestep_withlinvel \
num_envs=1 \
project_name=TestIsaacGymInstallation \
experiment_name=G123dof_loco \
headless=False
```

# Decoupled Locomotion
Here, we decouple the upper and lower body control in training. In detail, upper body motion is sampled from AMASS while the RL only controls the lower body. Therefore, you need to do the motion retargeting first before training.
## Motion Retargeting
Refer to [Notion](https://www.notion.so/Motion-Retargeting-1a01fa08c53580a38f3ee979734b2d1c?pvs=4).

## Lower-body RL Training with Upper Body IK

```bash
HYDRA_FULL_ERROR=1 python humanoidverse/train_agent.py \
+exp=decoupled_locomotion_stand_height \
+simulator=isaacgym \
+domain_rand=domain_rand_unitree_rl_gym_g1 \
+rewards=dec_loco/reward_g1_29dof_old_unitree_loco_stand_height \
+robot=g1/g1_29dof_old \
+terrain=terrain_locomotion_plane \
+obs=dec_loco/g1_29dof_obs_stand_height_history_wolinvel \
num_envs=4096 \
project_name=g1_29dof_dec_loco_stand \
experiment_name=unitree_NoPush_DR_rand_history_loco_stand_height_noise \
+opt=wandb \
obs.add_noise=True \
env.config.apply_waist_pitch_only_when_stance=True \
domain_rand.push_robots=False \
domain_rand.randomize_pd_gain=True \
robot.motion.reverse_motion=True \
robot.motion.motion_fps_scale=0.2 \
rewards.fix_upper_body=False \
rewards.reward_scales.penalty_action_rate=-0.25 \
rewards.reward_initial_penalty_scale=0.1 \
rewards.reward_penalty_degree=0.00002
```

## Multi-Agent WBC RL

```bash
HYDRA_FULL_ERROR=1 python humanoidverse/train_agent.py \
+exp=decoupled_locomotion_stand_height_waist_wbc_diff_force_ma_ppo_ma_env \
+simulator=isaacgym \
+domain_rand=domain_rand_unitree_rl_gym_g1 \
+rewards=dec_loco/reward_g1_29dof_waist_wbc_unitree_loco_stand_height_ma_diff_force_gait \
+robot=g1/g1_29dof_waist_fakehand \
+terrain=terrain_locomotion_plane \
+obs=dec_loco/g1_29dof_obs_stand_height_waist_wbc_force_history_wolinvel_ma_phase \
num_envs=4096 \
project_name=g1_29dof_wbc \
experiment_name=SenResWaistWBC_Loco_Diff_XYZ-Force_MAPPO_NewOBS_FricDR_Gait \
obs.add_noise=True \
+opt=wandb \
simulator.config.sim.physx.enable_dof_force_sensors=True \
env.config.zero_force_prob=[0.33,0.33,0.33] \
domain_rand.push_robots=False \
domain_rand.randomize_friction=True \
rewards.fix_upper_body=False \
rewards.reward_initial_penalty_scale=0.05 \
rewards.reward_penalty_degree=0.00003 \
rewards.upper_body_motion_scale_curriculum=False \
rewards.upper_body_tracking_sigma_curriculum=False \
env.config.residual_upper_body_action=True \
env.config.termination.terminate_when_low_upper_dof_tracking=False
```

# License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

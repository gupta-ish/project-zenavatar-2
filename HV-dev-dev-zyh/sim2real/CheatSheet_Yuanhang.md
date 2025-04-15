## Pre-Config
Here, we use `config/g1_29dof_free.yaml`. `free` means the robot base is free instead of fixed. Before testing sim2sim/sim2real, check the `INTERFACE` in the `.yaml` file:
```yaml
### REAL
# DOMAIN_ID: 0 # Domain id
# # Interface (Yuanhang: this can be commented out)
# INTERFACE: "en0"

## SIM
DOMAIN_ID: 0 # Domain id
# Interface (Yuanhang: this can NOT be commented out)
INTERFACE: "lo" # Yuanhang: in simulation, lo0 for mac, lo for linux
```

## Yuanhang: Dec Loco Stand Height & Force Estimation (G1_29DoF)

### Start Mujoco Env (ONLY for Sim2Sim)
```bash
# Linux
python sim_env/loco_manip.py --config=config/g1_29dof_free.yaml

python sim_env/loco_manip_tv.py --config=config/g1_29dof_free.yaml
# Mac
mjpython sim_env/loco_manip.py --config=config/g1_29dof_free.yaml

mjpython sim_env/loco_manip_tv.py --config=config/g1_29dof_free.yaml
```

### Launch the Policy
```bash

python rl_policy/decoupled_loco_manipulation.py --config=config/g1_29dof_free.yaml --model_path=./models/dec_loco/20250129_033843-unitree_NoPush_DR_rand_history_loco_stand_height_noise-decoupled_locomotion-g1_29dof/model_16800.onnx

python rl_policy/decoupled_loco_manipulation_force.py --config=config/g1_29dof_free_force.yaml --model_path=./models/dec_loco_force/20250212_164939-With_Force_Obs-decoupled_locomotion-g1_29dof_fakehand/model_44500.onnx

```

## Yuanhang: Dec Loco Stand Height Force (G1_29DoF)

### Start Mujoco Env (ONLY for Sim2Sim)
```bash
# Linux
python sim_env/loco_manip.py --config=config/g1_29dof_free_force.yaml

# Mac
mjpython sim_env/loco_manip.py --config=config/g1_29dof_free_force.yaml

```

### Launch the Policy
```bash

python rl_policy/decoupled_locomotion_stand_height_force.py --config=config/g1_29dof_free_force.yaml --model_path=./models/dec_loco_force/20250212_164939-With_Force_Obs-decoupled_locomotion-g1_29dof_fakehand/model_44500.onnx

python rl_policy/decoupled_locomotion_stand_height_force.py --config=config/g1_29dof_free_force.yaml --model_path=./models/dec_loco_force/20250212_200050-With_Random_Force_Obs-decoupled_locomotion-g1_29dof_fakehand/model_32200.onnx

python rl_policy/decoupled_locomotion_stand_height.py --config=config/g1_29dof_free.yaml --model_path=./models/dec_loco_force/20250212_024502-WO_Force_Obs-decoupled_locomotion-g1_29dof_fakehand/model_21000.onnx
```

## Yuanhang: Dec Loco Stand Height Waist Force (G1_29DoF)

### Start Mujoco Env (ONLY for Sim2Sim)
```bash
# Linux
python sim_env/loco_manip.py --config=config/g1_29dof_waist_free_force.yaml

# Mac
mjpython sim_env/loco_manip.py --config=config/g1_29dof_waist_free_force.yaml

```

### Launch the Policy
```bash

python rl_policy/dec_loco/decoupled_loco_manipulation_waist_force.py --config=config/g1_29dof_waist_free_force.yaml --model_path=./models/dec_loco_waist_force/20250305_202322-Waist_Force_Z-decoupled_locomotion-g1_29dof_fakehand/model_20000.onnx

```

## Yuanhang: Dec Loco Stand Height Waist Force Without Phase and Gait (G1_29DoF)

### Start Mujoco Env (ONLY for Sim2Sim)
```bash
# Linux
python sim_env/loco_manip.py --config=config/g1_29dof_waist_free_force_wophase.yaml

python sim_env/loco_manip.py --config=config/g1_29dof_waist_free_force_wophase_wocomp.yaml

# Mac
mjpython sim_env/loco_manip.py --config=config/g1_29dof_waist_free_force_wophase.yaml

```

### Launch the Policy
```bash

python rl_policy/dec_loco/decoupled_loco_manipulation_waist_force_wophase.py --config=config/g1_29dof_waist_free_force_wophase.yaml --model_path=./models/dec_loco_waist_force/20250312_145712-Waist_Diff_Force_Estimator-decoupled_locomotion-g1_29dof_fakehand/model_5700.onnx

python rl_policy/dec_loco/decoupled_loco_manipulation_waist_force_wophase.py --config=config/g1_29dof_waist_free_force_wophase.yaml --model_path=./models/dec_loco_waist_force/20250312_232306-Waist_Diff_Force_Estimator-decoupled_locomotion-g1_29dof_fakehand/model_7300.onnx

python rl_policy/dec_loco/decoupled_loco_manipulation_waist_force_wophase_wocomp.py --config=config/g1_29dof_waist_free_force_wophase_wocomp.yaml --model_path=./models/dec_loco_waist_force/20250313_051517-Waist_Diff_Force_Estimator_NoComp-decoupled_locomotion-g1_29dof_fakehand/model_10100.onnx

```
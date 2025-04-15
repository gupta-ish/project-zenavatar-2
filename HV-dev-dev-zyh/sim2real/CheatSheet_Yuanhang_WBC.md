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

## Yuanhang: WBC Loco Stand Height Waist Without Phase and Gait (G1_29DoF)

### Start Mujoco Env (ONLY for Sim2Sim)
```bash
# Linux
python sim_env/loco_manip.py --config=config/g1_29dof_free_wbc.yaml

# Mac
mjpython sim_env/loco_manip.py --config=config/g1_29dof_free_wbc.yaml

```

### Launch the Policy
```bash

python rl_policy/wbc_loco/wbc_loco_stand_height.py --config=config/g1_29dof_free_wbc.yaml --model_path=./models/wbc_loco/20250322_232048-ResWBC_Loco_Diff_XYZ-Force_MAPPO_NewOBS-decoupled_locomotion-g1_29dof_fakehand/model_31000.onnx

```

## Yuanhang: Waist WBC Loco Stand Height Waist Without Phase and Gait (G1_29DoF)

### Start Mujoco Env (ONLY for Sim2Sim)
```bash
# Linux
python sim_env/loco_manip.py --config=config/g1_29dof_waist_free_wbc.yaml

# Mac
mjpython sim_env/loco_manip.py --config=config/g1_29dof_waist_free_wbc.yaml

```

### Launch the Policy
```bash

python rl_policy/wbc_loco/wbc_loco_stand_height_waist.py --config=config/g1_29dof_waist_free_wbc.yaml --model_path=./models/wbc_loco/20250326_040136-ResWaistWBC_Loco_Diff_XYZ-Force_MAPPO_NewOBS_FricDR-decoupled_locomotion-g1_29dof_fakehand/model_21000.onnx

python rl_policy/wbc_loco/wbc_loco_stand_height_waist.py --config=config/g1_29dof_waist_free_wbc.yaml --model_path=./models/wbc_loco/20250325_030418-ResWaistWBC_Loco_Diff_XYZ-Force_MAPPO_NewOBS_FricDR-decoupled_locomotion-g1_29dof_fakehand/model_17600.onnx

python rl_policy/wbc_loco/wbc_loco_stand_height_waist.py --config=config/g1_29dof_waist_free_wbc.yaml --model_path=./models/wbc_loco/20250329_151509-SenResWaistWBC_Loco_Diff_XYZ-Force_MAPPO_NewOBS_FricDR-decoupled_locomotion-g1_29dof_fakehand/model_35000.onnx

python rl_policy/wbc_loco/wbc_loco_stand_height_waist.py --config=config/g1_29dof_waist_free_wbc.yaml --model_path=./models/wbc_loco/model_6100.onnx

python rl_policy/wbc_loco/wbc_loco_stand_height_waist.py --config=config/g1_29dof_waist_free_wbc.yaml --model_path=./models/wbc_loco/model_6100.onnx

python rl_policy/wbc_loco/wbc_loco_stand_height_waist.py --config=config/g1_29dof_waist_free_wbc.yaml --model_path=../logs/g1_29dof_wbc/20250331_193113-SenResWaistWBC_Loco_Diff_XYZ-Force_MAPPO_NewOBS_FricDR-decoupled_locomotion-g1_29dof_fakehand/exported/model_2900.onnx
```

## Yuanhang: Waist WBC Loco Stand Height Waist With Phase and Gait (G1_29DoF)

### Start Mujoco Env (ONLY for Sim2Sim)
```bash
# Linux
python sim_env/loco_manip.py --config=config/g1_29dof_waist_free_wbc_gait.yaml

# Mac
mjpython sim_env/loco_manip.py --config=config/g1_29dof_waist_free_wbc_gait.yaml

```

### Launch the Policy
```bash

python rl_policy/wbc_loco/wbc_loco_stand_height_waist_gait.py --config=config/g1_29dof_waist_free_wbc_gait.yaml --model_path=./models/wbc_loco/20250405_204846-SenResWaistWBC_Loco_Diff_XYZ-Force_MAPPO_NewOBS_FricDR_Gait-decoupled_locomotion-g1_29dof_fakehand/model_9400.onnx
```
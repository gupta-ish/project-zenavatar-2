## Yuanhang: Dec Loco Stand Height Waist Force Without Phase and Gait (G1_29DoF)

### Start Mujoco Env (ONLY for Sim2Sim)
```bash
# Linux

python sim_env/loco_manip.py --config=config/g1_29dof_waist_free_force_wophase_wocomp.yaml

# Mac
mjpython sim_env/loco_manip.py --config=config/g1_29dof_waist_free_force_wophase_wocomp.yaml

```

### Launch the Policy
```bash

python rl_policy/dec_loco/decoupled_loco_manipulation_waist_force_wophase_wocomp.py --config=config/g1_29dof_waist_free_force_wophase_wocomp.yaml --model_path=./models/dec_loco_waist_force/20250313_051517-Waist_Diff_Force_Estimator_NoComp-decoupled_locomotion-g1_29dof_fakehand/model_10100.onnx

```
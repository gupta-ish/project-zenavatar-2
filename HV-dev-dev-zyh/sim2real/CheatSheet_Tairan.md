## Tairan DeepMimic + Decoupled Locomotion (G1_29DoF, History)
1. Go to `sim2real` and activate `rvreal`:
    ```bash
    cd sim2real && mamba activate rvreal
    ```

2. If doing `Sim2Sim`:
    ```bash
    # Linux
    python sim_env/base_sim.py --config=config/g1_29dof_hist.yaml

    # Mac
    mjpython sim_env/base_sim.py --config=config/g1_29dof_hist.yaml
    ```

3. Publish `Low_State`:
    ```bash
    python state_publisher.py --config=config/g1_29dof_hist.yaml
    ```

4. Start your RL policy:
    Change 'rl_policy' to 'rl_inference' if you want to run tribble scripts 
    ```bash
    # Fixed Height No Delay
    python rl_policy/deepmimic_dec_loco.py --config=config/g1_29dof_hist.yaml --loco_model_path=./models/dec_loco/20241227_105729-noDR_rand_history_stand_trail1.0_ar0.5-decoupled_locomotion-g1_29dof/model_1300.onnx --mimic_model_paths=./models/mimic

    # Fixed Height With Delay
    python rl_policy/deepmimic_dec_loco.py --config=config/g1_29dof_hist.yaml --loco_model_path=./models/dec_loco/20250109_163529-noDR_rand_history_loco_stand-decoupled_locomotion-g1_29dof/exported/model_11200.onnx --mimic_model_paths=./models/mimic
    
    # Variable Height
    python rl_policy/deepmimic_dec_loco_height.py --config=config/g1_29dof_hist.yaml --loco_model_path=./models/dec_loco/20250109_231507-noDR_rand_history_loco_stand_height_noise-decoupled_locomotion-g1_29dof/model_6600.onnx --mimic_model_paths=./models/mimic

    # Variable Height with Large Friction DR
    python rl_policy/deepmimic_dec_loco_height.py --config=config/g1_29dof_hist.yaml --loco_model_path=./models/dec_loco/20250110_171530-unitreeDR_rand_history_loco_stand_height_noise-decoupled_locomotion-g1_29dof/exported/model_8000.onnx --mimic_model_paths=./models/mimic

    python rl_policy/deepmimic_dec_loco_height.py --config=config/g1_29dof_hist.yaml --loco_model_path=./models/model_53200.onnx --mimic_model_paths=./models/mimic
    ```

5. Send commands:
    ```bash
    python command_sender.py --config=config/g1_29dof_hist.yaml
    ```
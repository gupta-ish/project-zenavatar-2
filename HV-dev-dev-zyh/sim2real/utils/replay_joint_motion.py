import mujoco
import mujoco.viewer
import numpy as np
import csv
import time
import os
import pickle  # ← add this to handle .pkl saving

# === CONFIG ===
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(SCRIPT_DIR, "..", "..", "humanoidverse", "data", "robots", "g1", "g1_29dof_old_freebase_camera.xml")
CSV_FILES = [
    os.path.join(SCRIPT_DIR, "data", "recorded_joint_positions_1.csv"),
    os.path.join(SCRIPT_DIR, "data", "recorded_joint_positions_2.csv"),
    os.path.join(SCRIPT_DIR, "data", "recorded_joint_positions_3.csv"),
    os.path.join(SCRIPT_DIR, "data", "recorded_joint_positions_4.csv"),
]
NUM_SAMPLES_PER_FILE = 100
FPS = 10
SAVE_COMBINED_CSV_PATH = os.path.join(SCRIPT_DIR, "data", "combined_sampled_joint_positions.csv")
SAVE_COMBINED_PKL_PATH = os.path.join(SCRIPT_DIR, "data", "combined_sampled_joint_positions.pkl")

def load_and_sample_joint_data(csv_path, num_samples):
    data = np.loadtxt(csv_path, delimiter=",")
    if data.shape[0] < num_samples:
        raise ValueError(f"CSV at {csv_path} only has {data.shape[0]} rows; can't sample {num_samples} unique rows.")
    sampled_indices = np.random.choice(data.shape[0], size=num_samples, replace=False)
    return data[sampled_indices]

def main():
    # Step 1: Sample from all files and stack
    sampled_data_list = []
    for path in CSV_FILES:
        sampled_data = load_and_sample_joint_data(path, NUM_SAMPLES_PER_FILE)
        sampled_data_list.append(sampled_data)

    combined_data = np.vstack(sampled_data_list)
    np.random.shuffle(combined_data)  # Shuffle all 400 samples randomly
    print(f"Combined shape: {combined_data.shape}")  # Expect (400, 14)

    # Step 2: Save to CSV and PKL
    np.savetxt(SAVE_COMBINED_CSV_PATH, combined_data, delimiter=",", fmt="%.6f")
    print(f"Saved combined CSV to: {SAVE_COMBINED_CSV_PATH}")

    with open(SAVE_COMBINED_PKL_PATH, "wb") as f:
        pickle.dump(combined_data, f)
    print(f"Saved combined PKL to: {SAVE_COMBINED_PKL_PATH}")

    # Step 3: Load MuJoCo model
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Cannot find model: {MODEL_PATH}")

    model = mujoco.MjModel.from_xml_path(MODEL_PATH)
    data = mujoco.MjData(model)

    if model.nq < 29:
        raise ValueError(f"Expected at least 29 DoF, but model has {model.nq}")

    # Step 4: Replay in viewer
    with mujoco.viewer.launch_passive(model, data) as viewer:
        print(f"Replaying {combined_data.shape[0]} frames... Press ESC to exit.")
        for i in range(combined_data.shape[0]):
            qpos = np.zeros(model.nq)
            qpos[-14:] = combined_data[i]
            data.qpos[:] = qpos

            mujoco.mj_forward(model, data)
            viewer.sync()
            time.sleep(1 / FPS)

        print("Replay complete.")

if __name__ == "__main__":
    main()

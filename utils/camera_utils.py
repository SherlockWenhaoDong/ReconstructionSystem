import yaml
import numpy as np

def load_camera_parameters(yaml_file):
    with open(yaml_file, "r") as f:
        data = yaml.safe_load(f)

    # resolution
    width = data["calibration_resolution"]["width"]
    height = data["calibration_resolution"]["height"]

    # left parameters
    left_params = data["left"]["camera_matrix"]
    K_left = np.array([
        [left_params["fx"], 0, left_params["cx"]],
        [0, left_params["fy"], left_params["cy"]],
        [0, 0, 1]
    ])
    dist_left = np.array(data["left"]["dist_coefficients"])

    # right parameters
    right_params = data["right"]["camera_matrix"]
    K_right = np.array([
        [right_params["fx"], 0, right_params["cx"]],
        [0, right_params["fy"], right_params["cy"]],
        [0, 0, 1]
    ])
    dist_right = np.array(data["right"]["dist_coefficients"])

    # camera pose
    R = np.array(data["stereo"]["R"]).reshape(3, 3)
    t = np.array(data["stereo"]["t"]).reshape(3, 1)

    return {
        "resolution": (width, height),
        "K_left": K_left,
        "dist_left": dist_left,
        "K_right": K_right,
        "dist_right": dist_right,
        "R": R,
        "t": t
    }

if __name__ == "__main__":
    params = load_camera_parameters("camera.yaml")


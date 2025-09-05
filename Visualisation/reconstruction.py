import argparse
import os

import numpy as np
import cv2
import open3d as o3d

from DataProcessing.Dataloader import StereoFlowDataset
from Models.Depth import StereoDepth, StereoDepthSTTR, StereoDepthRAFT, EndoDACDepth, StereoDepthOpenCV, MonoDepthEndo
from Models.TSDF import GlobalTSDF
from utils.camera_utils import load_camera_parameters
from utils.pose_utils import warp, backproject, estimate_pose_pnp, pose_change_significant, Rt_to_pose, \
    rgbd_to_clean_pointcloud


def run_pipeline(left_dir, right_dir, camera_yaml, model_type, baseline_m=None,
                 voxel_size=0.004, sdf_trunc=0.02, show_every=5, flow_occ_th=1.0,
                 flow_conf_th=0.5, pnp_reproj_err=1.0, pnp_min_inliers=10):
    # camera
    cam = load_camera_parameters(camera_yaml)
    K = cam["K_left"]
    R = cam["R"]
    t = cam["t"]

    if baseline_m is None:
        # try to infer baseline from stereo t (units seem in meters? If t is large, use it directly)
        baseline = float(np.linalg.norm(cam["t"]))  # note: original yaml may be in mm; adjust if needed
    else:
        baseline = baseline_m

    # dataset
    ds = StereoFlowDataset(left_dir, right_dir)
    n = len(ds)
    print(f"Dataset loaded: {n} frames")

    # models (replace with real inference)
    if model_type == 'raftstereo':
        stereo = StereoDepth('/home/dongwenhao/SurgicalRecon/Models/raftstereo-realtime.pth')
    elif model_type == 'STTR':
        stereo = StereoDepthSTTR('/home/dongwenhao/SurgicalRecon/Models/sttr_light_sceneflow_pretrained_model.pth.tar')
    elif model_type == 'RAFT':
        stereo = StereoDepthRAFT('/home/dongwenhao/SurgicalRecon/Models/raft-things.pth')
    elif model_type == 'EndoDAC':
        stereo = EndoDACDepth('/home/dongwenhao/SurgicalRecon/Models/depth_model.pth')
    elif model_type == 'EndoOmni':
        stereo = MonoDepthEndo('/home/dongwenhao/SurgicalRecon/Models/EndoOmni_bf.pt')
    else:
        stereo = StereoDepthOpenCV()


    tsdf = GlobalTSDF(voxel_size=voxel_size, sdf_trunc=sdf_trunc)
    poses_world = []  # world <- cam0

    # visualizer setup
    # vis = o3d.visualization.Visualizer()
    # vis.create_window("Realtime TSDF", width=1280, height=720)
    # pcd_vis = o3d.geometry.PointCloud()
    # coord = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.05)
    # vis.add_geometry(coord)
    # vis.add_geometry(pcd_vis)

    prev_sample = None
    T_w_prev = Rt_to_pose(R, t)
    changed_frame = []

    if '1' in right_dir:
        global_path = f"exp1/{model_type}/Global/"
        cur_path = f"exp1/{model_type}/Current/"
        all_path = f"exp1/{model_type}/"
    else:
        global_path = f"exp2/{model_type}/Global/"
        cur_path = f"exp2/{model_type}/Current/"
        all_path = f"exp2/{model_type}/"
    os.makedirs(all_path, exist_ok=True)
    os.makedirs(global_path, exist_ok=True)
    os.makedirs(cur_path, exist_ok=True)


    for idx in range(n):
        print(f'Processing frame-{idx}!')
        sample = ds[idx]
        left_img = sample["left"]["img"]
        right_img = sample["right"]["img"]
        left_mask = sample["left"]["mask"]
        # flows: may be None if missing
        lf_flow = sample["left"]["fwd_flow"]
        lf_mask = sample["left"]["fwd_mask"]
        lb_flow = sample["left"]["bwd_flow"]
        lb_mask = sample["left"]["bwd_mask"]
        dist_left = np.array(cam['dist_left']).reshape(-1, 1)
        dist_right = np.array(cam['dist_right']).reshape(-1, 1)

        if lf_flow is None:
            lf_flow = np.zeros((left_img.shape[0], left_img.shape[1], 2), dtype=np.float32)
            lf_mask = np.zeros((left_img.shape[0], left_img.shape[1]), dtype=np.uint8)
        if lb_flow is None:
            lb_flow = np.zeros((left_img.shape[0], left_img.shape[1], 2), dtype=np.float32)
            lb_mask = np.zeros((left_img.shape[0], left_img.shape[1]), dtype=np.uint8)

        # compute occlusion mask via forward-backward
        occ = warp(lf_flow, lb_flow, flow_occ_th)

        # segmentation mask (tools). dataset mask may be None -> treat as all background
        if left_mask is None:
            seg_mask = np.zeros((left_img.shape[0], left_img.shape[1]), dtype=np.uint8)
        else:
            seg_mask = left_mask.astype(np.uint8)

        # compute depth: use stereo model (placeholder) -> disparity -> depth (meters)
        if model_type == 'EndoOmni' or model_type == 'EndoDAC':
            disp_left = stereo.disparity(np.array(left_img))
            disp_right = stereo.disparity(np.array(right_img))
            disp = stereo.calibrate_left_disp(disp_left, disp_right, K[0, 0], baseline)
        else:
            disp = stereo.disparity(np.array(left_img), np.array(right_img))  # (H,W) in pixels
        depth = (K[0,0] * baseline) / (disp + 1e-8)       # Z = f*B / disp
        min_val = np.min(depth)
        max_val = np.max(depth)
        if max_val - min_val < 1e-6:
            return np.zeros_like(depth)
        depth = (depth - min_val) / (max_val - min_val)
        depth[disp <= 0] = 0.0

        # build valid mask (use flow mask from npz and occ and seg)
        valid_flow_mask = (lf_mask.astype(np.bool_) if lf_mask is not None else np.ones_like(occ))
        valid_mask = (~occ) & (~seg_mask.astype(bool)) & valid_flow_mask & (depth > 0)

        # if enough valid points, run PnP to compute relative pose from t -> t+1
        T_rel = None
        inliers_count = 0
        if prev_sample is not None:
            ys, xs = np.where(valid_mask)
            if xs.shape[0] >= pnp_min_inliers:
                res = estimate_pose_pnp(lf_flow, depth, K, seg_mask, conf_th=flow_conf_th, pnp_reproj_err=pnp_reproj_err, min_inliers=pnp_min_inliers)
                if res is not None:
                    T_rel, inliers = res
                    inliers_count = len(inliers)
        if T_rel is None:
            T_w_curr = T_w_prev.copy()
        else:
            # composition: world <- cam_t1 = world <- cam_t @ cam_t -> cam_t1
            if pose_change_significant(T_rel):
                # update world poses: T_w_prev is world <- cam_t
                T_w_prev = poses_world[-1]
                T_w_curr = T_w_prev @ T_rel
                print('Camera Pose Changed!')
                changed_frame.append(idx)

        poses_world.append(T_w_curr)

        # integrate current left frame into TSDF using static region mask
        static_mask = (~occ) & (~seg_mask.astype(bool)) & (depth > 0)
        depth_masked = depth * static_mask.astype(np.float32)
        # integrate into TSDF (pass world<-cam)
        tsdf.integrate(cv2.cvtColor(np.array(left_img), cv2.COLOR_RGB2BGR), depth_masked, K, T_w_curr)

        # visualization update every show_every frames
        if (idx % show_every) == 0 or idx == n - 1:
            pcd = tsdf.extract_pointcloud()
            points = np.asarray(pcd.points)
            colors = np.asarray(pcd.colors)
            colors = np.clip(colors, 0.0, 1.0)
            pcd_cur = rgbd_to_clean_pointcloud(np.array(left_img), depth_masked, K)
            if len(np.asarray(pcd.points)) > 0:
                pcd.points = o3d.utility.Vector3dVector(points)
                pcd.colors = o3d.utility.Vector3dVector(colors)

                print(f"Saved {len(points)} points with valid colors.")


                filename = global_path + f"tsdf_frame_{idx:05d}.ply"
                curfilename = cur_path + f"frame_{idx:05d}.ply"
                o3d.io.write_point_cloud(filename, pcd)
                o3d.io.write_point_cloud(curfilename, pcd_cur)
                print(f"[{idx:05d}] integrated frame; pose_inliers={inliers_count}; saved {filename} and {curfilename}")


        prev_sample = sample

    # final export
    mesh = tsdf.extract_mesh()
    pcd_final = tsdf.extract_pointcloud()
    o3d.io.write_triangle_mesh(all_path + "scene_mesh.ply", mesh)
    o3d.io.write_point_cloud(all_path + "scene_cloud.ply", pcd_final)
    print("Saved scene_mesh.ply and scene_cloud.ply")
    print(f"Camera Pose changed {len(changed_frame)} times!")


# -------------------------
# CLI
# -------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Stereo-flow TSDF reconstruction (no-resize)")
    parser.add_argument("--left", required=True, default='/home/dongwenhao/SurgicalRecon/frames/L_clip2', help="Left dataset root")
    parser.add_argument("--right", required=True, default='/home/dongwenhao/SurgicalRecon/frames/R_clip2', help="Right dataset root")
    parser.add_argument("--camera", required=True, default='/home/dongwenhao/SurgicalRecon/Stereo_Skilltest/camera_params.yaml', help="Camera YAML file path")
    parser.add_argument("--baseline", type=float, default=None, help="baseline in meters (overrides YAML)")
    parser.add_argument("--voxel", type=float, default=0.005, help="TSDF voxel size (m)")
    parser.add_argument("--trunc", type=float, default=0.01, help="TSDF truncation (m)")
    parser.add_argument("--show-every", type=int, default=10, help="visualize every N frames")
    parser.add_argument("--flow-occ-th", type=float, default=1.0, help="forward-backward occ threshold (px)")
    parser.add_argument("--flow-conf-th", type=float, default=0.5, help="flow confidence threshold (unused if using dataset masks)")
    parser.add_argument("--pnp-reproj", type=float, default=500.0, help="PnP reprojection error (px)")
    parser.add_argument("--pnp-min-inliers", type=int, default=100, help="PnP min inliers")
    parser.add_argument("--depth_model", type=str, default="raftstereo", help="model name for depth prediction")

    args = parser.parse_args()

    run_pipeline(
        args.left, args.right, args.camera, baseline_m=args.baseline,
        voxel_size=args.voxel, sdf_trunc=args.trunc,
        show_every=args.show_every, flow_occ_th=args.flow_occ_th,
        flow_conf_th=args.flow_conf_th, pnp_reproj_err=args.pnp_reproj,
        pnp_min_inliers=args.pnp_min_inliers, model_type=args.depth_model
    )
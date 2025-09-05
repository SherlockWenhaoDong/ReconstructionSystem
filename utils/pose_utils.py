import os
import glob
import cv2
import numpy as np
from typing import Dict, Tuple, List
import open3d as o3d


def warp(flow_fw, flow_bw, flow_occ_th):
    """
    Compute forward-backward consistency occlusion mask.

    Behavior:
      - Accepts numpy.ndarray or torch.Tensor for flow_fw and flow_bw (H,W,2).
      - If one of the flows is missing or is a placeholder (e.g. shape (1,1,2)),
        the function cannot compute forward-backward consistency and returns
        an all-False mask (no occlusion).
      - Returns a boolean array of shape (H,W), True = occluded / inconsistent.

    Args:
      flow_fw: forward flow (t -> t+1), numpy or torch tensor, shape (H,W,2) or None/placeholder
      flow_bw: backward flow (t+1 -> t), numpy or torch tensor, shape (H,W,2) or None/placeholder
      flow_occ_th: threshold in pixels for forward-backward consistency

    Returns:
      occ: np.ndarray of dtype bool with shape (H, W)
    """
    shapes = []
    if isinstance(flow_fw, np.ndarray) and flow_fw.ndim == 3:
        shapes.append(flow_fw.shape[:2])
    if isinstance(flow_bw, np.ndarray) and flow_bw.ndim == 3:
        shapes.append(flow_bw.shape[:2])

    if len(shapes) == 0:
        raise ValueError("Both flow_fw and flow_bw are None or invalid; cannot infer image size.")

    shapes_sorted = sorted(shapes, key=lambda s: s[0]*s[1], reverse=True)
    H, W = shapes_sorted[0]

    # helper to check if a flow is usable (has matching H,W and 3 dims)
    def usable_flow(f):
        return isinstance(f, np.ndarray) and f.ndim == 3 and f.shape[0] == H and f.shape[1] == W and f.shape[2] == 2

    if not usable_flow(flow_fw) or not usable_flow(flow_bw):
        return np.zeros((H, W), dtype=bool)

    flow_fw = flow_fw.astype(np.float32)
    flow_bw = flow_bw.astype(np.float32)

    gx, gy = np.meshgrid(np.arange(W), np.arange(H))
    x_fw = gx + flow_fw[..., 0]
    y_fw = gy + flow_fw[..., 1]

    def bilinear_sample(flow, x, y):
        x0 = np.floor(x).astype(np.int32); x1 = x0 + 1
        y0 = np.floor(y).astype(np.int32); y1 = y0 + 1

        x0 = np.clip(x0, 0, W - 1); x1 = np.clip(x1, 0, W - 1)
        y0 = np.clip(y0, 0, H - 1); y1 = np.clip(y1, 0, H - 1)

        Ia = flow[y0, x0]; Ib = flow[y1, x0]; Ic = flow[y0, x1]; Id = flow[y1, x1]

        wa = (x1 - x) * (y1 - y)
        wb = (x1 - x) * (y - y0)
        wc = (x - x0) * (y1 - y)
        wd = (x - x0) * (y - y0)

        wa = wa.astype(np.float32); wb = wb.astype(np.float32)
        wc = wc.astype(np.float32); wd = wd.astype(np.float32)

        return Ia * wa[..., None] + Ib * wb[..., None] + Ic * wc[..., None] + Id * wd[..., None]

    bw_on_fw = bilinear_sample(flow_bw, x_fw, y_fw)
    diff = flow_fw + bw_on_fw
    mag = np.linalg.norm(diff, axis=-1)

    occ = mag > flow_occ_th
    return occ

def load_frame_pairs(left_dir:str, right_dir:str) -> List[Tuple[str,str]]:
    """ Load sorted left-right image pairs """
    ls = sorted(glob.glob(os.path.join(left_dir, "*.png")))
    rs = sorted(glob.glob(os.path.join(right_dir, "*.png")))
    pairs = []
    for l, r in zip(ls, rs):
        pairs.append((l, r))
    return pairs

def disp_to_depth(disp: np.ndarray, K: np.ndarray, baseline: float) -> np.ndarray:
    """ Convert disparity to depth """
    f = K[0, 0]
    with np.errstate(divide='ignore'):
        depth = f * baseline / (disp + 1e-8)
    depth[disp <= 0] = 0
    return depth

def backproject(u, v, z, K):
    """ Backproject 2D pixel coordinates (u,v) with depth z to 3D points """
    fx, fy, cx, cy = K[0,0], K[1,1], K[0,2], K[1,2]
    X = (u - cx) * z / fx
    Y = (v - cy) * z / fy
    return np.stack([X, Y, z], axis=-1)

def rodrigues_to_rotvec(R):
    """ Convert rotation matrix to rotation vector (Rodrigues) """
    rvec, _ = cv2.Rodrigues(R)
    return rvec.reshape(-1)

def rotmat_from_rvec(rvec):
    """ Convert rotation vector to rotation matrix """
    R, _ = cv2.Rodrigues(rvec.reshape(3,1))
    return R

def angle_between_rot(R):
    """ Compute rotation angle (degrees) from rotation matrix """
    tr = np.trace(R)
    val = (tr - 1.0) / 2.0
    val = np.clip(val, -1.0, 1.0)
    return np.degrees(np.arccos(val))

def compose_pose(T_prev, T_rel):
    """ Compose previous pose with relative pose """
    return T_prev @ T_rel

def invert_pose(T):
    """ Invert a 4x4 pose matrix """
    R = T[:3,:3]
    t = T[:3,3]
    Tinv = np.eye(4)
    Tinv[:3,:3] = R.T
    Tinv[:3,3]  = -R.T @ t
    return Tinv

def estimate_pose_pnp(flow_fw: np.ndarray,
                      depth_t: np.ndarray,
                      K: np.ndarray,
                      seg_mask_t: np.ndarray,
                      conf=1,
                      conf_th=0.5,
                      pnp_reproj_err=3.0,
                      min_inliers=50):
    """
    Use depth at frame t + forward flow t->t+1 to construct 3D-2D correspondences
    and solve PnP for T_t->t+1
    """
    h, w = depth_t.shape
    yy, xx = np.where((depth_t > 0) & (conf > conf_th) & (seg_mask_t == 0))
    if len(xx) < min_inliers:
        return None

    u0 = xx.astype(np.float32)
    v0 = yy.astype(np.float32)
    z  = depth_t[yy, xx].astype(np.float32)

    pts3d = backproject(u0, v0, z, K).astype(np.float32)

    u1 = u0 + flow_fw[yy, xx, 0].astype(np.float32)
    v1 = v0 + flow_fw[yy, xx, 1].astype(np.float32)
    pts2d = np.stack([u1, v1], axis=-1).astype(np.float32)

    ok, rvec, tvec, inliers = cv2.solvePnPRansac(
        pts3d, pts2d, K, None,
        reprojectionError=pnp_reproj_err,
        iterationsCount=300, confidence=0.999,
        flags=cv2.SOLVEPNP_ITERATIVE
    )
    if not ok or inliers is None or len(inliers) < min_inliers:
        return None

    rvec, tvec = cv2.solvePnPRefineLM(
        pts3d[inliers[:,0]], pts2d[inliers[:,0]], K, None, rvec, tvec
    )

    R, _ = cv2.Rodrigues(rvec)
    t = tvec.reshape(3)
    T = np.eye(4, dtype=np.float32)
    T[:3,:3] = R
    T[:3, 3] = t
    return T, inliers


def pose_change_significant(T_rel: np.ndarray)->bool:
    rot_deg_threshold = 0.8
    trans_m_threshold = 0.005
    R = T_rel[:3,:3]
    t = T_rel[:3,3]
    ang = angle_between_rot(R)
    trans = np.linalg.norm(t)
    return (ang > rot_deg_threshold) or (trans > trans_m_threshold)

def Rt_to_pose(R, t):
    """
    R: 3x3 rotation matrix
    t: 3x1 translation vector
    returns 4x4 pose matrix
    """
    T = np.eye(4, dtype=np.float32)
    T[:3, :3] = R
    T[:3, 3] = t.flatten()
    return T

def rgbd_to_clean_pointcloud(rgb, depth, K, max_depth=1, black_threshold=30):
    """
    Convert RGBD image into a cleaned point cloud (suitable for endoscopy)

    Args:
        rgb: (H, W, 3) RGB image, dtype=uint8
        depth: (H, W) depth map (float32/float64), in meters (or mm if consistent with intrinsics)
        K: (3, 3) camera intrinsic matrix
        max_depth: maximum depth to keep (removes far points)
        black_threshold: threshold for filtering out nearly black pixels (0-255)

    Returns:
        o3d.geometry.PointCloud: processed point cloud
    """
    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]

    H, W = depth.shape
    u, v = np.meshgrid(np.arange(W), np.arange(H))

    Z = depth.flatten()
    X = (u.flatten() - cx) * Z / fx
    Y = (v.flatten() - cy) * Z / fy
    points = np.vstack((X, Y, Z)).T

    colors = rgb.reshape(-1, 3)

    valid_depth = (Z > 0) & (Z < max_depth)

    valid_color = np.any(colors > black_threshold, axis=1)

    mask = valid_depth & valid_color
    points = points[mask]
    colors = colors[mask] / 255.0

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.colors = o3d.utility.Vector3dVector(colors)

    if len(points) > 0:
        pcd = pcd.voxel_down_sample(voxel_size=0.005)
        pcd, _ = pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)

    return pcd




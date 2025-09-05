import os
import glob
import cv2
import numpy as np
from typing import Dict, Tuple, List
import open3d as o3d


class GlobalTSDF:
    def __init__(self, voxel_size, sdf_trunc):
        self.voxel_size = voxel_size
        self.sdf_trunc = sdf_trunc
        self.tsdf = o3d.pipelines.integration.ScalableTSDFVolume(
            voxel_length=voxel_size,
            sdf_trunc=sdf_trunc,
            color_type=o3d.pipelines.integration.TSDFVolumeColorType.RGB8
        )
    def integrate(self, color_bgr: np.ndarray, depth_m: np.ndarray, K: np.ndarray, T_w_c: np.ndarray):
        h, w = depth_m.shape
        color_rgb = cv2.cvtColor(color_bgr, cv2.COLOR_BGR2RGB)
        color_o3d = o3d.geometry.Image(color_rgb.astype(np.uint8))
        depth_o3d = o3d.geometry.Image(depth_m.astype(np.float32))
        rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
            color_o3d, depth_o3d,
            depth_scale=1.0, depth_trunc=100,
            convert_rgb_to_intensity=False
        )
        intrinsic = o3d.camera.PinholeCameraIntrinsic(
            width=w, height=h, fx=K[0,0], fy=K[1,1], cx=K[0,2], cy=K[1,2]
        )

        self.tsdf.integrate(rgbd, intrinsic, T_w_c.astype(np.float64))

    def extract_pointcloud(self):
        return self.tsdf.extract_point_cloud()

    def extract_mesh(self):
        m = self.tsdf.extract_triangle_mesh()
        m.compute_vertex_normals()
        return m


import os

import cv2
import torch
from pathlib import Path
import numpy as np

from Models.EndoOmni.models.depth_anything.dpt import DepthAnything
from Models.RAFT_Stereo.core.raft_stereo import RAFTStereo
from Models.RAFT_Stereo.core.utils.utils import InputPadder
from Models.STTR.dataset.preprocess import compute_left_occ_region
from Models.STTR.module.sttr import STTR
from Models.STTR.utilities.misc import NestedTensor
from Models.RAFT.raft import RAFT
from argparse import Namespace
import torch.nn.functional as F

DEVICE = 'cuda'

class StereoDepth:
    """Compute disparity from left/right images using RAFT-Stereo."""
    def __init__(self, ckpt: str, device: str = "cuda"):
        self.device = device

        import argparse
        args = argparse.Namespace(
            restore_ckpt=ckpt,
            hidden_dims=[128]*3,
            corr_implementation="alt",
            shared_backbone=False,
            corr_levels=4,
            corr_radius=4,
            n_downsample=2,
            context_norm="batch",
            slow_fast_gru=False,
            n_gru_layers=3,
            valid_iters=32,
            mixed_precision=False
        )

        self.model = RAFTStereo(args)

        state_dict = torch.load(ckpt, map_location=device)
        if 'sceneflow' in ckpt:
            new_state_dict = {k.replace("Module.", ""): v for k, v in state_dict.items()}
        elif 'realtime' in ckpt:
            new_state_dict = {k.replace("update_block.", ""): v for k, v in state_dict.items()}
        else:
            new_state_dict = state_dict
        self.model.load_state_dict(new_state_dict, strict=False)

        self.model.to(self.device)
        self.model.eval()

    def disparity(self, left_bgr: np.ndarray, right_bgr: np.ndarray) -> np.ndarray:
        """Compute disparity from left/right BGR images (HxWx3)."""
        import torch

        left = torch.from_numpy(left_bgr).permute(2,0,1).float().to(self.device).unsqueeze(0)
        right = torch.from_numpy(right_bgr).permute(2,0,1).float().to(self.device).unsqueeze(0)

        padder = InputPadder(left.shape, divis_by=32)
        left, right = padder.pad(left, right)

        with torch.no_grad():
            _, flow_up = self.model(left, right, iters=32, test_mode=True)

        flow_up = padder.unpad(flow_up).squeeze()  # remove batch dim
        disp = -flow_up.cpu().numpy()
        return disp

    def forward(self, left_bgr: np.ndarray, right_bgr: np.ndarray):
        """fine_tune depth model."""
        import torch

        left = torch.from_numpy(left_bgr).permute(2, 0, 1).float().to(self.device).unsqueeze(0)
        right = torch.from_numpy(right_bgr).permute(2, 0, 1).float().to(self.device).unsqueeze(0)

        padder = InputPadder(left.shape, divis_by=32)
        left, right = padder.pad(left, right)

        _, flow_up = self.model(left, right, iters=32, test_mode=True)

        disp = -padder.unpad(flow_up).squeeze()  # remove batch dim

        return disp


class StereoDepthSTTR:
    """Compute disparity from left/right images using STTR."""
    def __init__(self, ckpt: str, device: str = "cuda"):
        self.device = device

        import argparse
        args = argparse.Namespace(
            channel_dim=128,
            position_encoding='sine1d_rel',
            num_attn_layers=6,
            nheads=4,
            regression_head='ot',
            context_adjustment_layer='cal',
            cal_num_blocks=8,
            cal_feat_dim=16,
            cal_expansion_ratio=4
        )

        self.model = STTR(args).to(self.device).eval()

        checkpoint = torch.load(ckpt, map_location=device)
        pretrained_dict = checkpoint['state_dict']
        pretrained_dict = {k.replace("tokenizer.", ""): v for k, v in pretrained_dict.items()}
        self.model.load_state_dict(pretrained_dict, strict=False)
        print("✅ Pre-trained STTR model successfully loaded.")

    def disparity(self, left_bgr: np.ndarray, right_bgr: np.ndarray) -> np.ndarray:
        """
        Compute disparity from left/right BGR images (HxWx3).
        Input: left_bgr, right_bgr -> numpy arrays
        Output: disp -> numpy disparity map (H,W)
        """
        left = torch.from_numpy(left_bgr).permute(2,0,1).float()[None] / 255.0
        right = torch.from_numpy(right_bgr).permute(2,0,1).float()[None] / 255.0

        sample = {
            'left': left,
            'right': right,
        }
        sample = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v for k, v in sample.items()}
        nested = NestedTensor(sample['left'], sample['right'])

        with torch.no_grad():
            disp_pred = self.model(nested)['disp_pred']

        disp = disp_pred.squeeze().cpu().numpy()
        return disp

    def forward(self, left_bgr: np.ndarray, right_bgr: np.ndarray):
        """fine_tune depth model."""
        left = torch.from_numpy(left_bgr).permute(2, 0, 1).float()[None] / 255.0
        right = torch.from_numpy(right_bgr).permute(2, 0, 1).float()[None] / 255.0

        sample = {
            'left': left,
            'right': right,
        }
        sample = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v for k, v in sample.items()}
        nested = NestedTensor(sample['left'], sample['right'])

        with torch.no_grad():
            disp_pred = self.model(nested)['disp_pred']

        disp = disp_pred.squeeze().cpu().numpy()
        return disp


class StereoDepthRAFT:
    """Compute disparity from left/right stereo images using standard RAFT."""

    def __init__(self, ckpt: str, device: str = "cuda"):
        """
        Args:
            ckpt (str): Path to RAFT checkpoint (.pth)
            device (str): "cuda" or "cpu"
        """
        self.device = device

        args = Namespace(
            small=False,
            mixed_precision=False,
            alternate_corr=False,
        )
        self.model = RAFT(args).to(self.device).eval()

        checkpoint = torch.load(ckpt, map_location=self.device)
        if "state_dict" in checkpoint:
            checkpoint = checkpoint["state_dict"]
        self.model.load_state_dict(checkpoint, strict=False)
        print("✅ RAFT model loaded from", ckpt)

    def _preprocess(self, img: np.ndarray) -> torch.Tensor:
        """
        Convert HxWx3 uint8 numpy image -> 1x3xHxW float tensor [0,1].
        """
        ten = torch.from_numpy(img).permute(2, 0, 1).float()[None] / 255.0
        return ten.to(self.device)

    def disparity(self, left_bgr: np.ndarray, right_bgr: np.ndarray):
        """
        Compute disparity from left/right stereo images using RAFT.

        Args:
            left_bgr (np.ndarray): left image (H,W,3), dtype=uint8
            right_bgr (np.ndarray): right image (H,W,3), dtype=uint8

        Returns:
            disp (np.ndarray): disparity map (H,W)
            mask (np.ndarray): valid mask (H,W)
        """
        left = self._preprocess(left_bgr)
        right = self._preprocess(right_bgr)
        padder = InputPadder(left.shape)
        left, right = padder.pad(left, right)

        with torch.no_grad():
            flow_fw = self.model(left, right, iters=20, test_mode=True)[1]
            flow_bw = self.model(right, left, iters=20, test_mode=True)[1]

        flow_fw = padder.unpad(flow_fw)
        flow_bw = padder.unpad(flow_bw)

        flow_fw = flow_fw[0].permute(1, 2, 0).cpu().numpy()
        flow_bw = flow_bw[0].permute(1, 2, 0).cpu().numpy()



        disp = self._flow_to_disp(flow_fw, flow_bw)
        return disp

    def _flow_to_disp(self, forward_flow, backward_flow):
        """Convert RAFT flows to disparity with optional consistency check."""
        H, W, _ = forward_flow.shape
        disparity = -forward_flow[..., 0]
        return disparity




class EndoDACDepth:
    """
    Compute disparity from endoscopic RGB image using EndoDAC pre-trained model.
    """

    def __init__(self, ckpt: str, device: str = "cuda"):
        """
        Args:
            ckpt (str): Path to EndoDAC checkpoint (pretrained weights).
            device (str): Device to load model on, e.g. "cuda" or "cpu".
        """
        self.device = device


        import Models.EndoDAC.models.endodac as endodac
        depther_dict = torch.load(ckpt)
        args = Namespace(
            image_path="/home/dongwenhao/SurgicalRecon/test_images",
            model_path="/home/dongwenhao/SurgicalRecon/checkpoints/endodac.pth",
            ext="png",
            no_cuda=False,
            pretrained_path="/home/dongwenhao/SurgicalRecon/Models/",
            lora_rank=4,
            lora_type="dvlora",
            residual_block_indexes=[2, 5, 8, 11],
            include_cls_token=True,
            model_type="endodac"
        )
        self.model = endodac.endodac(
            backbone_size="base", r=args.lora_rank, lora_type=args.lora_type,
            image_shape=(224, 280), pretrained_path=args.pretrained_path,
            residual_block_indexes=args.residual_block_indexes,
            include_cls_token=args.include_cls_token)
        model_dict = self.model.state_dict()
        self.model.load_state_dict({k: v for k, v in depther_dict.items() if k in model_dict})
        self.model.cuda()
        self.model.eval()
        print(f"✅ EndoDAC model loaded from {ckpt}")

    def _preprocess(self, img: np.ndarray) -> torch.Tensor:
        img = img.astype(np.float32) / 255.0
        tensor = torch.from_numpy(img).permute(2, 0, 1)[None].to(self.device)
        return tensor

    def disparity(self, img: np.ndarray) -> np.ndarray:
        """
        Predict disparity map from a single endoscopic RGB image.

        Args:
            img (np.ndarray): Input RGB image (H, W, 3), uint8.

        Returns:
            disp (np.ndarray): Estimated disparity map (H, W), float32.
        """
        img = self._preprocess(img)
        with torch.no_grad():
            outputs = self.model(img)

        disp = outputs[("disp", 0)]
        disp_resized = torch.nn.functional.interpolate(
            disp, (img.shape[-2], img.shape[-1]), mode="bilinear", align_corners=False)
        disp = disp_resized.squeeze().cpu().numpy().astype(np.float32)
        return disp

    def calibrate_left_disp(self, disp_L, disp_R, f, baseline):
        """
        Calibrate left-view disparity using the right-view disparity.

        Args:
            disp_L (np.ndarray): Predicted left-view disparity map (H, W)
            disp_R (np.ndarray): Predicted right-view disparity map (H, W)
            f (float): Camera focal length
            baseline (float): Stereo camera baseline distance

        Returns:
            np.ndarray: Calibrated left-view disparity map (H, W)
        """
        H, W = disp_L.shape
        disp_L_calib = np.zeros_like(disp_L)

        u, v = np.meshgrid(np.arange(W), np.arange(H))

        u_R = (u - disp_L).astype(np.int32)
        u_R = np.clip(u_R, 0, W - 1)

        Z_R = f * baseline / (disp_R[v, u_R] + 1e-8)

        disp_L_calib = f * baseline / (Z_R + 1e-8)

        disp_L_final = (disp_L + disp_L_calib) / 2.0

        return disp_L_final


class StereoDepthOpenCV:
    """
    Compute disparity from left/right RGB images using OpenCV StereoSGBM.
    """

    def __init__(self, min_disp=0, num_disp=128, block_size=5):
        """
        Args:
            min_disp (int): Minimum disparity (default 0)
            num_disp (int): Maximum disparity minus minimum disparity, must be divisible by 16
            block_size (int): Matching block size (default 5)
        """
        self.min_disp = min_disp
        self.num_disp = num_disp
        self.block_size = block_size

        self.stereo = cv2.StereoSGBM_create(
            minDisparity=self.min_disp,
            numDisparities=self.num_disp,
            blockSize=self.block_size,
            P1=8 * 3 * self.block_size ** 2,
            P2=32 * 3 * self.block_size ** 2,
            disp12MaxDiff=1,
            uniquenessRatio=10,
            speckleWindowSize=100,
            speckleRange=32
        )

    def disparity(self, left_img, right_img):
        """
        Compute disparity map from left/right RGB images.

        Args:
            left_img (np.ndarray): Left image, HxWx3, BGR or RGB
            right_img (np.ndarray): Right image, HxWx3, BGR or RGB

        Returns:
            np.ndarray: Disparity map (H, W), float32
        """
        if left_img.ndim == 3:
            left_gray = cv2.cvtColor(left_img, cv2.COLOR_BGR2GRAY)
        else:
            left_gray = left_img
        if right_img.ndim == 3:
            right_gray = cv2.cvtColor(right_img, cv2.COLOR_BGR2GRAY)
        else:
            right_gray = right_img

        disp = self.stereo.compute(left_gray, right_gray).astype(np.float32) / 16.0

        disp[disp < 0] = 0.0

        return disp



class MonoDepthEndo:
    """
    Wrapper class for single-image endoscopic depth estimation.
    Provides methods to predict depth or disparity from a single RGB image.
    """

    def __init__(self, model_path, model_type="DPT_DINOv2", device="cuda", encoder="vitb"):
        """
        Args:
            model_path (str): Path to pretrained weights
            model_type (str): Model type, default 'DPT_DINOv2' (EndoDAC)
            device (str): 'cuda' or 'cpu'
            encoder (str): Encoder backbone, e.g., 'vitb'
        """
        self.device = torch.device(device)
        self.model = self._load_model(model_path, model_type, encoder)
        self.model.to(self.device)
        self.model.eval()

    def _load_model(self, model_path, model_type, encoder):
        """
        Load the model with given weights.
        """
        if model_type == "DPT_DINOv2":
            model = DepthAnything(encoder=encoder)
            checkpoint = torch.load(model_path, map_location=self.device)
            # Some checkpoints store state dict under 'model'
            if "model" in checkpoint:
                model.load_state_dict(checkpoint["model"])
            else:
                model.load_state_dict(checkpoint)
        else:
            raise NotImplementedError(f"Model type '{model_type}' not implemented")
        return model

    def pad_image(self, img, multiple=14):
        """Pad image so that H, W are multiples of 'multiple'."""
        h, w = img.shape[:2]
        pad_h = (multiple - h % multiple) % multiple
        pad_w = (multiple - w % multiple) % multiple

        padded = np.pad(img, ((0, pad_h), (0, pad_w), (0, 0)), mode='constant')
        return padded, (h, w)

    def crop_to_original(self, pred, orig_size):
        """Crop prediction back to original image size."""
        h, w = orig_size
        return pred[:h, :w]

    @torch.no_grad()
    def disparity(self, img):
        """
        Compute disparity map from a single image (left or right).
        img: np.ndarray (H, W, 3), RGB image
        return: np.ndarray (H, W), disparity map
        """
        img_padded, orig_size = self.pad_image(img, 14)

        img_tensor = torch.from_numpy(img_padded).permute(2, 0, 1).float().unsqueeze(0).to(self.device) / 255.0

        pred = self.model(img_tensor)
        if isinstance(pred, (list, tuple)):
            pred = pred[-1]
        elif isinstance(pred, dict):
            pred = pred.get('metric_depth', pred.get('out'))

        pred = pred.squeeze().cpu()

        pred = F.interpolate(pred.unsqueeze(0).unsqueeze(0),
                             size=img_padded.shape[:2],
                             mode="bilinear",
                             align_corners=False).squeeze().numpy()

        pred = self.crop_to_original(pred, orig_size)

        return pred

    def calibrate_left_disp(self, disp_L, disp_R, f, baseline):
        """
        Calibrate left-view disparity using the right-view disparity.

        Args:
            disp_L (np.ndarray): Predicted left-view disparity map (H, W)
            disp_R (np.ndarray): Predicted right-view disparity map (H, W)
            f (float): Camera focal length
            baseline (float): Stereo camera baseline distance

        Returns:
            np.ndarray: Calibrated left-view disparity map (H, W)
        """
        H, W = disp_L.shape
        disp_L_calib = np.zeros_like(disp_L)

        u, v = np.meshgrid(np.arange(W), np.arange(H))

        u_R = (u - disp_L).astype(np.int32)
        u_R = np.clip(u_R, 0, W - 1)

        Z_R = f * baseline / (disp_R[v, u_R] + 1e-8)

        disp_L_calib = f * baseline / (Z_R + 1e-8)

        disp_L_final = (disp_L + disp_L_calib) / 2.0

        return disp_L_final



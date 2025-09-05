import os
import glob
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms


class StereoFlowDataset(Dataset):
    def __init__(self, left_dir, right_dir, transform=None):
        """
        Args:
            left_dir: path to left dataset folder
            right_dir: path to right dataset folder
            transform: torchvision transforms for images (default: ToTensor)
        """
        self.left_dir = left_dir
        self.right_dir = right_dir

        self.left_img_dir = os.path.join(left_dir, "images")
        self.right_img_dir = os.path.join(right_dir, "images")

        self.left_img_files = sorted(glob.glob(os.path.join(self.left_img_dir, "*.png")))
        self.right_img_files = sorted(glob.glob(os.path.join(self.right_img_dir, "*.png")))

        assert len(self.left_img_files) == len(self.right_img_files), \
            "Left and right image counts do not match!"

        self.num_img = len(self.left_img_files)


    def __len__(self):
        return self.num_img

    def load_image(self, path):
        img = Image.open(path).convert("RGB")
        return img

    def load_mask(self, base_dir, filename):
        mask_file = os.path.join(base_dir, "masks", filename)
        if not os.path.exists(mask_file):
            mask = np.zeros((1, 1), dtype=np.uint8)
        else:
            mask = np.array(Image.open(mask_file).convert("L"))
        return (mask > 0).astype(np.float32)

    def load_flow(self, base_dir, idx, direction="fwd"):
        """
        Load flow .npz file (with 'flow' and 'mask' keys).
        direction: "fwd" or "bwd"
        """
        if direction == "fwd" and idx == self.num_img - 1:
            flow, mask = np.zeros((1, 1, 2), np.float32), np.zeros((1, 1), np.float32)
        elif direction == "bwd" and idx == 0:
            flow, mask = np.zeros((1, 1, 2), np.float32), np.zeros((1, 1), np.float32)
        else:
            flow_file = os.path.join(base_dir, "flow", f"{idx:05d}_{direction}.npz")
            data = np.load(flow_file)
            flow = data["flow"]
            mask = data["mask"]

        return flow, mask

    def __getitem__(self, idx):
        filename = os.path.basename(self.left_img_files[idx])

        # === Left view ===
        left_img = self.load_image(self.left_img_files[idx])
        left_mask = self.load_mask(self.left_dir, filename)
        left_fwd_flow, left_fwd_mask = self.load_flow(self.left_dir, idx, "fwd")
        left_bwd_flow, left_bwd_mask = self.load_flow(self.left_dir, idx, "bwd")

        # === Right view ===
        right_img = self.load_image(self.right_img_files[idx])
        right_mask = self.load_mask(self.right_dir, filename)
        right_fwd_flow, right_fwd_mask = self.load_flow(self.right_dir, idx, "fwd")
        right_bwd_flow, right_bwd_mask = self.load_flow(self.right_dir, idx, "bwd")

        return {
            "left": {
                "img": left_img,
                "mask": left_mask,
                "fwd_flow": left_fwd_flow,
                "fwd_mask": left_fwd_mask,
                "bwd_flow": left_bwd_flow,
                "bwd_mask": left_bwd_mask,
            },
            "right": {
                "img": right_img,
                "mask": right_mask,
                "fwd_flow": right_fwd_flow,
                "fwd_mask": right_fwd_mask,
                "bwd_flow": right_bwd_flow,
                "bwd_mask": right_bwd_mask,
            },
            "name": filename
        }


# Example usage
if __name__ == "__main__":
    dataset = StereoFlowDataset(
        left_dir="/home/dongwenhao/SurgicalRecon/frames/L_clip1",
        right_dir="/home/dongwenhao/SurgicalRecon/frames/R_clip1",
    )
    sample = dataset[0]
    print("Left img:", sample["left"]["img"].shape,
          "Right img:", sample["right"]["img"].shape,
          "Flow:", sample["left"]["fwd_flow"].shape)

from PIL import Image
from matplotlib import pyplot as plt

import torch
from torchvision import transforms
from torch.autograd import Variable
import Models.pytorch_segmentation_detection.models.resnet_dilated as resnet_dilated

import numpy as np
import os

# Paths
img_path = '/home/dongwenhao/SurgicalRecon/frames/L_clip1/images'
mask_path = '/home/dongwenhao/SurgicalRecon/frames/L_clip1/masks'
os.makedirs(mask_path, exist_ok=True)

# Get sorted list of image files
files = os.listdir(img_path)
files.sort()

# Image preprocessing transforms
valid_transform = transforms.Compose([
    transforms.CenterCrop((1008, 1264)),  # Center crop to original resolution
    # Note: This resize distorts image non-isotropically, used just to measure time correctly
    transforms.Resize(size=(840, 1250)),
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
])

# Load segmentation model
fcn = resnet_dilated.Resnet9_8s(num_classes=2)
fcn.load_state_dict(torch.load('/home/dongwenhao/SurgicalRecon/Models/resnet_9_8s.pth'))
fcn.cuda()
fcn.eval()

# Loop through all images and generate masks
for fname in files:
    print(f'Processing {fname}...')
    img_file = os.path.join(img_path, fname)

    # Open image
    img_not_preprocessed = Image.open(img_file).convert('RGB')
    img = valid_transform(img_not_preprocessed)
    img = img.unsqueeze(0)  # Add batch dimension
    img = Variable(img.cuda())

    # Inference
    with torch.no_grad():
        res = fcn(img)
        _, tmp = res.squeeze(0).max(0)
        segmentation = tmp.data.cpu().numpy().astype(np.uint8)

    # Save segmentation mask
    mask_file = os.path.join(mask_path, fname)  # Keep same name as original
    mask_img = Image.fromarray(segmentation * 255)  # Convert to 0/255 grayscale
    mask_img.save(mask_file)

print("✅ All segmentation results have been saved to:", mask_path)

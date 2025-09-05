# ReconstructionSystem
This repository provides the code for Wenhao's system.

# Data Preprocessing
1. Dataloader.py
   Here I define the data loader. How do I read the dataset during the reconstruction process?
2. Flow_generator.py
   This is used to generate optical flow for interframe analysis.
3. Frame_cutter.py
   This is used for frame cropping in video processing.
4. Mask_generator.py
   This file generates a mask for each frame using a pre-trained model.
   

# Models
1. EndoDAC
   EndoDAC Source Code for Depth Estimation
2. EndoOmni
   EndoOmni Source Code for Depth Estimation
3. pytorch_segmentation_detection
   Segementation Network Source Code
4. RAFT
   RAFT Source Code
5. RAFT_Stereo
   RAFT_Stereo Source Code for Depth Estimation
6. STTR(stereoTransformer)
   STTR Source Code for Depth Estimation
7. Depth.py
   This defines how to read the pre-trained model of the aforementioned model for deep estimation.
8. TSDF.py
   This defines how RGBD information is saved as TSDF and updated.
# utils
1. camera_utils.py
   Read camera parameters
2. flow_utils.py
   How to handle optical flow and its essential operations, as referenced from DynamicNeRF.
3. pose_util.py
   Camera pose-related operations and how to estimate camera pose.

# Visualisation
1. reconstruction.py
   Rebuild the system's main function and select the deep model you wish to use from the rebuild parameter list for deep prediction.


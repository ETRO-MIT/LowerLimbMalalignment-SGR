# Python packages
import os
import torch
from pathlib import Path

# Custom functions
from Models.BaseModels import UNet16
from Models.SGRModel import SGRNetwork16
from Functions.UtilsSGR import execute_knee_transformation, execute_femur_transformation, execute_ankle_transformation


# Class that builds an SGR model depending on the type of joint to be analyzed: hip (femur), knee, or ankle. After
# building the model, detection of the landmarks takes place. Finally, post-processing is done to pass the estimated
# landmarks (that are on the 512 x 512 coordinate system) to the original coordinate system (the one of the FLL X-ray).
class DetectionSGR:
    def __init__(self, type_joint: str, main_working_directory: Path):
        super(DetectionSGR, self).__init__()
        # Set device
        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

        # According to the type of joint, set the following parameters: path to the weights, number of coordinates to
        # estimate, and the U-Net model that will be included on the segmentation branch.
        # For the Ankle and Knee models, the last layer of the U-Net is modified to yield extra masks (corresponding to
        # the number of landmarks to be estimated).
        self.type_joint = type_joint
        # Ankle
        if self.type_joint == 'Ankle':
            self.path_to_weights = os.path.join(main_working_directory,
                                                'Weights/Ankle/Best_Run_ep100_bs8_lr1e-05_a0.2.pth')
            self.number_of_coordinates = 4  # 2 x-y landmarks

            # Create U-Net -> Modify last layer to output two masks
            self.unet = UNet16()
            self.unet.final = torch.nn.Conv2d(in_channels=32, out_channels=2, kernel_size=(1, 1), stride=(1, 1))

        # Femur
        elif self.type_joint == 'Femur':
            self.path_to_weights = os.path.join(main_working_directory,
                                                'Weights/Femur/Best_Run_ep100_bs8_lr1e-05_a0.2.pth')
            self.number_of_coordinates = 2  # 1 x-y landmark
            self.unet = UNet16()

        # Knee
        elif self.type_joint == 'Knee':
            self.path_to_weights = os.path.join(main_working_directory,
                                                'Weights/Knee/Best_Run_ep100_bs8_lr1e-05_a0.2.pth')
            self.number_of_coordinates = 10  # 5 x-y landmarks

            # Create U-Net -> Modify last layer to output two masks
            self.unet = UNet16()
            self.unet.final = torch.nn.Conv2d(in_channels=32, out_channels=5, kernel_size=(1, 1), stride=(1, 1))

        # Error
        else:
            raise ValueError('Select a correct type of joint: "Ankle", "Femur", or "Knee".')

    # Build SGR model
    def build_model(self):
        # Create SGR model
        sgr_model = SGRNetwork16(unet16=self.unet, num_coordinates=self.number_of_coordinates)
        sgr_model.load_state_dict(torch.load(self.path_to_weights, map_location=self.device))
        sgr_model.to(self.device)

        return sgr_model

    # Make estimations
    def estimate_landmarks(self, torch_xrays_joints, sitk_xrays_joints, sitk_fll_xray):
        # Un-pack the data
        torch_right_xray_joint, torch_left_xray_joint = torch_xrays_joints[0], torch_xrays_joints[1]
        sitk_right_xray_joint, sitk_left_xray_joint = sitk_xrays_joints[0], sitk_xrays_joints[1]

        # Build the model
        sgr_model = self.build_model()

        # Pass the torch images to format [Bs, Chn, Width, Height]
        torch_right_xray_joint = torch.unsqueeze(torch_right_xray_joint, dim=0).to(self.device)
        torch_left_xray_joint = torch.unsqueeze(torch_left_xray_joint, dim=0).to(self.device)

        # Estimate the landmarks
        sgr_model.eval()
        with torch.no_grad():
            # Estimate position of the landmarks
            _, right_landmarks = sgr_model(torch_right_xray_joint)
            _, left_landmarks = sgr_model(torch_left_xray_joint)

        # Transform the estimations to the original coordinate system. Physical points are needed for calculating
        # the misalignment metrics, the pixel coordinates are required for visualization purposes.
        if self.type_joint == 'Ankle':
            px_512, px_fll, phys_points_fll = execute_ankle_transformation(right_landmarks, left_landmarks,
                                                                           sitk_right_xray_joint,
                                                                           sitk_left_xray_joint,
                                                                           sitk_fll_xray)
        elif self.type_joint == 'Femur':
            px_512, px_fll, phys_points_fll = execute_femur_transformation(right_landmarks, left_landmarks,
                                                                           sitk_right_xray_joint,
                                                                           sitk_left_xray_joint,
                                                                           sitk_fll_xray)
        elif self.type_joint == 'Knee':
            px_512, px_fll, phys_points_fll = execute_knee_transformation(right_landmarks, left_landmarks,
                                                                          sitk_right_xray_joint,
                                                                          sitk_left_xray_joint,
                                                                          sitk_fll_xray)
        else:
            raise ValueError('Unknown joint type')

        return px_512, px_fll, phys_points_fll

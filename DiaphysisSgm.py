# Python packages
import os
import torch

# Custom functions
from Models.BaseModels import UNet16V2 as UNet16
from Functions.Diaphysis import get_diaphysis_lines, post_process_filter


# Class that builds a U-Net for the segmentation of the diaphysis bone. After the mask is obtained, post-processing is
# applied to obtain the required landmark: center of the diaphysis. This landmark is needed for the calculation of the
# FVA angle.
class DiaphysisSgm:
    def __init__(self, main_directory):
        super(DiaphysisSgm, self).__init__()
        # Set default parameters
        self.path_to_weights = os.path.join(main_directory, 'Weights/Diaphysis/Best_DiceCELoss_lr1e5.pth')
        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    # Build the U-Net model
    def build_model(self):
        unet = UNet16()
        unet.load_state_dict(torch.load(self.path_to_weights, map_location=self.device))
        unet.to(self.device)

        return unet

    # Estimate the bone mask
    def make_estimation(self, torch_diaphysis_xrays, sitk_diaphysis_xrays, sitk_fll_xray):
        # Un-pack the data
        torch_image_right, torch_image_left = torch_diaphysis_xrays[0], torch_diaphysis_xrays[1]
        sitk_image_right, sitk_image_left = sitk_diaphysis_xrays[0], sitk_diaphysis_xrays[1]

        # Build the U-Net model
        unet_model = self.build_model()

        # Pass the torch images to format [Bs, Chn, Width, Height]
        torch_image_right = torch.unsqueeze(torch_image_right, dim=0).to(self.device)
        torch_image_left = torch.unsqueeze(torch_image_left, dim=0).to(self.device)

        # Execute estimation
        unet_model.eval()
        with torch.no_grad():
            # Estimate masks
            right_diaphysis_mask = unet_model(torch_image_right)
            left_diaphysis_mask = unet_model(torch_image_left)

            # Post-processing of the diaphysis masks
            post_right_diaphysis = post_process_filter(right_diaphysis_mask)
            post_left_diaphysis = post_process_filter(left_diaphysis_mask)

        # Get diaphysis landmarks (pixel and physical point coordinates)
        diaphysis_right_phys_point, diaphysis_left_phys_point = get_diaphysis_lines(sitk_image_right,
                                                                                    post_right_diaphysis,
                                                                                    sitk_image_left,
                                                                                    post_left_diaphysis)
        diaphysis_right_pix_point = sitk_fll_xray.TransformPhysicalPointToIndex(diaphysis_right_phys_point)
        diaphysis_left_pix_point = sitk_fll_xray.TransformPhysicalPointToIndex(diaphysis_left_phys_point)

        # Pack data
        diaphysis_pix_points = [diaphysis_right_pix_point, diaphysis_left_pix_point]
        diaphysis_phys_points = [diaphysis_right_phys_point, diaphysis_left_phys_point]

        return diaphysis_pix_points, diaphysis_phys_points

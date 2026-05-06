# Python packages
import torch
import numpy as np
import SimpleITK as sitk

# Custom functions
from .PreProcessing import get_new_spacing, resample_image


# Open morphological operation
def open_filter(sitk_image):
    sitk_open = sitk.BinaryMorphologicalOpeningImageFilter()
    sitk_open.SetKernelRadius(5)
    sitk_open.SetKernelType(sitk.sitkBall)

    return sitk_open.Execute(sitk_image)


# Post-process filter to generate a more smooth mask
def post_process_filter(torch_prediction):
    sigmoid = torch.nn.Sigmoid()
    pred_sigmoid = sigmoid(torch_prediction)
    array_prediction = (pred_sigmoid[0, 0, :, :] > 0.5).cpu().numpy().astype('uint8')
    sitk_prediction = sitk.GetImageFromArray(array_prediction)
    sitk_prediction_post = open_filter(sitk_prediction)

    return sitk_prediction_post


# Get middle point of the last mask line
def get_physical_points_from_mask(sitk_xray, sitk_mask):
    # Pass to array
    array_mask = sitk.GetArrayFromImage(sitk_mask)

    # Get last row where mask is present
    rr, cc = np.where(array_mask == 1)
    array_last_row = array_mask[rr[-1], :]

    # Get values of the first and last edge
    edge_indexes = np.where(array_last_row == 1)
    right_edge_idx = edge_indexes[0][0]
    left_edge_idx = edge_indexes[0][-1]

    # Coordinates of the points
    right_point = [int(right_edge_idx), int(rr[-1])]
    left_point = [int(left_edge_idx), int(rr[-1])]

    # Physical points
    right_physical_point = sitk_xray.TransformIndexToPhysicalPoint(right_point)
    left_physical_point = sitk_xray.TransformIndexToPhysicalPoint(left_point)

    return right_physical_point, left_physical_point


# Get diaphysis line points for each image -> Return the order version of it
def get_diaphysis_lines(diaphysis_xray1, diaphysis_mask1, diaphysis_xray2, diaphysis_mask2):
    # Pair xray1
    size1 = diaphysis_xray1.GetSize()
    origin1 = diaphysis_xray1.GetOrigin()

    # Read Mask1 and resample to original dimensions
    diaphysis_msk1_new_spacing = get_new_spacing(diaphysis_mask1, new_size=[size1[0], size1[1]])
    diaphysis_msk1_resampled = resample_image(diaphysis_mask1,
                                             out_spacing=[diaphysis_msk1_new_spacing[0], diaphysis_msk1_new_spacing[1]],
                                             is_label=True)
    diaphysis_msk1_resampled.CopyInformation(diaphysis_xray1)

    # Pair xray2
    size2 = diaphysis_xray2.GetSize()
    origin2 = diaphysis_xray2.GetOrigin()

    # Read Mask2 and resample to original dimensions
    diaphysis_msk2_new_spacing = get_new_spacing(diaphysis_mask2, new_size=[size2[0], size2[1]])
    diaphysis_msk2_resampled = resample_image(diaphysis_mask2,
                                             out_spacing=[diaphysis_msk2_new_spacing[0], diaphysis_msk2_new_spacing[1]],
                                             is_label=True)
    diaphysis_msk2_resampled.CopyInformation(diaphysis_xray2)

    # Define right and left femur
    x1 = origin1[0]
    x2 = origin2[0]
    x_half = (x1 + x2) / 2
    if x1 < x_half:
        diaphysis_right_xray = diaphysis_xray1
        diaphysis_right_mask = diaphysis_msk1_resampled
        diaphysis_left_xray = diaphysis_xray2
        diaphysis_left_mask = diaphysis_msk2_resampled
    else:
        diaphysis_right_xray = diaphysis_xray2
        diaphysis_right_mask = diaphysis_msk2_resampled
        diaphysis_left_xray = diaphysis_xray1
        diaphysis_left_mask = diaphysis_msk1_resampled

    # Get coordinates of the lines
    diaphysis_phys_right_line = get_physical_points_from_mask(diaphysis_right_xray,
                                                                                       diaphysis_right_mask)
    diaphysis_phys_left_line = get_physical_points_from_mask(diaphysis_left_xray,
                                                                                     diaphysis_left_mask)

    # Get mid-points
    right_phys_mid_point = [(diaphysis_phys_right_line[0][0] + diaphysis_phys_right_line[1][0]) / 2,
                            (diaphysis_phys_right_line[0][1] + diaphysis_phys_right_line[1][1]) / 2]
    left_phys_mid_point = [(diaphysis_phys_left_line[0][0] + diaphysis_phys_left_line[1][0]) / 2,
                           (diaphysis_phys_left_line[0][1] + diaphysis_phys_left_line[1][1]) / 2]

    return right_phys_mid_point, left_phys_mid_point

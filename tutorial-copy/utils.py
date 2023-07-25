import torch
import random
import numpy as np

def vertical_boundary_tensor_image():
    vertical_boundary_image = [[]] 
    boundary_position = random.randint(1, 255)
    for pixel_row in range(256):
        vertical_boundary_image[0].append([])
        for pixel_column in range(256):
            if pixel_column < boundary_position:
                vertical_boundary_image[0][pixel_row].append(np.float32(1))
            else:
                vertical_boundary_image[0][pixel_row].append(np.float32(0))

    vertical_boundary_tensor = torch.tensor(vertical_boundary_image)
    return vertical_boundary_tensor

def horizontal_boundary_tensor_image():
    horizontal_boundary_image = [[]] 
    boundary_position = random.randint(1, 255)
    for pixel_row in range(256):
        horizontal_boundary_image[0].append([])
        if pixel_row < boundary_position:
            for pixel_column in range(256):
                horizontal_boundary_image[0][pixel_row].append(np.float32(1))
        else:
            for pixel_column in range(256):
                horizontal_boundary_image[0][pixel_row].append(np.float32(0))

    horizontal_boundary_tensor = torch.tensor(horizontal_boundary_image)
    return horizontal_boundary_tensor
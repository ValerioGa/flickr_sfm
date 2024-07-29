
"""
This code takes a distorted image and undistorts it.
The image should be converted to a torch.tensor for the undistortion.
"""

import torch
import torchvision
from PIL import Image
import torch.nn.functional as F
import os
from tqdm import tqdm

os.chdir('..')
os.chdir('..')

input_dir ='/home/vgallo/flickr_images_nogeo/quicktestdist'
output_dir = '/home/vgallo/flickr_images_nogeo/quicktest'
    
for im_name in tqdm(os.listdir(input_dir)):
    # Open the distorted image with PIL
    img = Image.open(os.path.join(input_dir, im_name))
    #img.show()

    # Convert PIL image to torch.tensor
    tensor = torchvision.transforms.ToTensor()(img)
    assert tensor.shape == torch.Size([3, 512, 512])
    # The tensor has shape [3, 512, 512], we need to add a dimension at the beginning
    tensor = tensor.reshape(1, 3, 512, 512)

    # Some cool functions to undistort your image
    undistortion_tensor = torch.load("IMCvenv/sfm/grid.torch")
    tensor = F.grid_sample(tensor, undistortion_tensor)

    # Remove the extra dimension
    tensor = tensor.reshape(3, 512, 512)
    # Convert back to PIL image, so we can visualize it
    img = torchvision.transforms.ToPILImage()(tensor)
    #img.show()
    if not os.path.exists(output_dir):
            os.mkdir(output_dir)

    img.save(os.path.join(output_dir, im_name))
    


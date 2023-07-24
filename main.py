from torch.utils.data import Dataset
import os
from PIL import Image
from torchvision import transforms

class FolderDataset(Dataset):
    """
    Creates a PyTorch dataset from folder, returning two tensor images.
    Args: 
    main_dir : directory where images are stored.
    transform (optional) : torchvision transforms to be applied while making dataset
    """

    def __init__(self, main_dir, transform=None):
        self.main_dir = main_dir
        self.transform = transform
        self.all_imgs = os.listdir(main_dir)

    def __len__(self):
        return len(self.all_imgs)

    def __getitem__(self, idx):
        img_loc = os.path.join(self.main_dir, self.all_imgs[idx])
        image = Image.open(img_loc).convert("L")

        if self.transform is not None:
            tensor_image = self.transform(image)

        return tensor_image, tensor_image

images_dataset = FolderDataset("images", transform=transforms.ToTensor())
image_to_show = 0

print(f"\n\n{len(images_dataset)} images in the dataset")
print(f"images_dataset is of type {type(images_dataset)}\n\n")

print(f"Each element of the dataset is a {type(images_dataset[image_to_show])}")
print(f"...with a length of {len(images_dataset[image_to_show])}\n\n")

print(f"Each element of the tuple is a {type(images_dataset[image_to_show][0])}\n\n")
print(f"The tensor values are (themselves vectors, tensors??): [channels, vertical_pixels, horizontal_pixels]: {images_dataset[image_to_show][0].shape}\n\n")

world_dataset = FolderDataset("simple-images", transform=transforms.ToTensor())
world_image = world_dataset[0][0]

def renderPixel(value):
    if value < 0.25:
        return chr(32)
    elif value < 0.5:
        return chr(46)
    elif value < 0.75:
        return chr(111)
    else:
        return chr(88)
    

print(world_image.shape)
print(world_image[0, 0, 0].item())

for row in world_image[0]:
    for pixel in row:
        print(renderPixel(pixel.item()), end="")
    print()




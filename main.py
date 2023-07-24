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
        image = Image.open(img_loc).convert("RGB")

        if self.transform is not None:
            tensor_image = self.transform(image)

        return tensor_image, tensor_image

images_dataset = FolderDataset("images", transform=transforms.ToTensor())
image_to_show = 2

print(f"\n\n{len(images_dataset)} images in the dataset\n\n")
print(f"images_dataset is of type {type(images_dataset)}")

print(f"Each element of the dataset is a {type(images_dataset[image_to_show])}\n\n")

print(f"Each element of the tuple is a {type(images_dataset[image_to_show][0])}\n\n")
print(f"The tensor values are (themselves vectors, tensors??): [channels, vertical_pixels, horizontal_pixels]: {images_dataset[0][0].shape}\n\n")

print("All images seem to have been converted to the same size - as the smallest image? How did this happen? Are larger images being cropped?")


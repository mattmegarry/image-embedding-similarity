from torchvision import transforms

images_dataset = FolderDataset("images", transform=transforms.ToTensor())
image_to_show = 0

print(f"\n\n{len(images_dataset)} images in the dataset")
print(f"images_dataset is of type {type(images_dataset)}\n\n")

print(f"Each element of the dataset is a {type(images_dataset[image_to_show])}")
print(f"...with a length of {len(images_dataset[image_to_show])}\n\n")

print(f"Each element of the tuple is a {type(images_dataset[image_to_show][0])}\n\n")
print(f"The tensor values are (themselves vectors, tensors??): [channels, vertical_pixels, horizontal_pixels]: {images_dataset[image_to_show][0].shape}\n\n")
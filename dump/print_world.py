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
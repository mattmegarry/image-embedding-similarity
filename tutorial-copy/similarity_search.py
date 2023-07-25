from image_compute_similar import compute_similar_images
from PIL import Image
import numpy as np
import glob

from image_similarity_dataset import FolderDataset

def main():
    embedding = np.load('data_embedding.npy')
    image = Image.open('../images/obama_1.jpg').convert('L')
    indices_list = compute_similar_images(image, 10, embedding)
    dataset = FolderDataset('../images/')
    for i in indices_list[0]:
        print(dataset.all_imgs[i])

if __name__ == '__main__':
    main()
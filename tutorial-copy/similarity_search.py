from image_compute_similar import compute_similar_images
from PIL import Image
import numpy as np
import glob

from utils import vertical_boundary_tensor_image, horizontal_boundary_tensor_image 

from image_similarity_dataset import FolderDataset

def main():
    embedding = np.load('data_embedding.npy')
    image = Image.open('../face-images/obama_1.jpg').convert('L')
    image = horizontal_boundary_tensor_image()
    indices_list = compute_similar_images(image, 10, embedding)
    dataset = FolderDataset('../face-images/')
    for i in indices_list[0]:
        print(dataset.all_imgs[i])

    

if __name__ == '__main__':
    main()

""" 

    embedding = np.load('data_embedding.npy')
    one_h_image = horizontal_boundary_tensor_image()
    one_v_image = vertical_boundary_tensor_image()
    h_predictions = compute_similar_images(one_h_image, 5, embedding)
    v_predictions = compute_similar_images(one_v_image, 5, embedding)
    print(h_predictions)
    print(v_predictions)

 """
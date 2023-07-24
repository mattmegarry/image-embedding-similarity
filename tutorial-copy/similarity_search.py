from image_compute_similar import compute_similar_images
from PIL import Image
import numpy as np

def main():
    embedding = np.load('data_embedding.npy')
    image = Image.open('../images/thurman_1.jpg')
    indices_list = compute_similar_images(image, 20, embedding)
    print(indices_list)

if __name__ == '__main__':
    main()
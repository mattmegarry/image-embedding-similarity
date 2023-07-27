import torch
import torchvision.transforms as T

from sklearn.neighbors import NearestNeighbors
from image_similarity_encoder_model import ConvEncoder

def compute_similar_images(image, num_images, embedding):
    """
    Given an image and number of similar images to search.
    Returns the num_images closest neares images.
    Args:
    image: Image whose similar images are to be found.
    num_images: Number of similar images to find.
    embedding : A (num_images, embedding_dim) Embedding of images learnt from auto-encoder.
    device : "cuda" or "cpu" device.
    """

    # Load pretrained encoder 
    encoder = ConvEncoder()
    encoder.load_state_dict(torch.load("encoder_model.pt"))
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    encoder.to(device)
        
    image_tensor = T.ToTensor()(image)
    """ image_tensor = image """
    image_tensor = image_tensor.unsqueeze(0) 


    
    with torch.no_grad():
        image_embedding = encoder(image_tensor).cpu().detach().numpy()
        
    flattened_embedding = image_embedding.reshape((image_embedding.shape[0], -1))

    knn = NearestNeighbors(n_neighbors=num_images, metric="cosine")
    knn.fit(embedding)

    _, indices = knn.kneighbors(flattened_embedding)
    indices_list = indices.tolist()
    return indices_list
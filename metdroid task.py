import cv2
from facenet_pytorch import InceptionResnetV1, MTCNN
import torch
from scipy.spatial.distance import euclidean
import numpy as np

class FaceRecognition:
    """
    A class for face recognition that utilizes MTCNN for face detection 
    and InceptionResnetV1 for generating face embeddings. It also computes 
    the Euclidean distance between face embeddings from two images.
    """

    def __init__(self, device='cpu'):
        """
        Initializes the face recognition model.

        Parameters:
            device (str): The device to run the model on ('cpu' or 'cuda'). Default is 'cpu'.
        """
        self.device = torch.device(device)  # Set the device (CPU or GPU)
        self.mtcnn = MTCNN(keep_all=True, device=self.device)  # Initialize MTCNN face detector
        self.resnet = InceptionResnetV1(pretrained='vggface2').eval().to(self.device)  # Initialize face embedding model
    
    def extract_embedding_from_image(self, image_path):
        """
        Extracts a face embedding from a single image. Prints the number of face embeddings detected.

        Parameters:
            image_path (str): The file path of the image to process.

        Returns:
            numpy.ndarray: The flattened 1D face embedding, or None if no face is detected.
        """
        # Read and process the image
        image = cv2.imread(image_path)
        imagergb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Detect faces in the image
        boxes, probs = self.mtcnn.detect(imagergb)

        if boxes is not None:
            # Extract faces using MTCNN
            faces = self.mtcnn(imagergb)  # This returns a tensor of detected faces

            if faces is not None:
                # Print the number of faces (embeddings) detected
                print(f"Number of embeddings (faces) detected: {len(faces)}")

                # Assume the first detected face (if any) is the one to use
                face_tensor = faces[0]
                # Get the embedding for the face
                embedding = self.resnet(face_tensor.unsqueeze(0)).detach().cpu().numpy()  # Add batch dimension and get the embedding
                return embedding.flatten()  # Return the flattened embedding as a 1D vector
            else:
                print(f"No faces found in {image_path}.")
                return None
        else:
            print(f"No faces detected in {image_path}.")
            return None

    def compute_distance_between_images(self, image_path1, image_path2):
        """
        Computes the Euclidean distance between the embeddings of two images.

        Parameters:
            image_path1 (str): The file path of the first image.
            image_path2 (str): The file path of the second image.

        Returns:
            float: The Euclidean distance between the two face embeddings, or None if no faces are detected in either image.
        """
        # Extract embeddings for both images
        embedding1 = self.extract_embedding_from_image(image_path1)
        embedding2 = self.extract_embedding_from_image(image_path2)

        if embedding1 is not None and embedding2 is not None:
            # Compute the Euclidean distance between the two embeddings
            distance = euclidean(embedding1, embedding2)
            return distance
        else:
            print("Could not compute distance due to missing face embeddings.")
            return None

# Example usage:
if __name__ == "__main__":
    face_recognition = FaceRecognition(device='cpu')

    # Paths to the images
    image_path1 = r"path_to_first_image"
    image_path2 = r"path_to_second_image"

    # Compute and print the Euclidean distance between the two images
    distance = face_recognition.compute_distance_between_images(image_path1, image_path2)
    if distance is not None:
        print(f"Euclidean distance between the two face embeddings: {distance}")

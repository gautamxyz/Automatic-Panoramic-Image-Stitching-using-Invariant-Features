import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

class featureExtraction:
    def __init__(self,image_list,k=2,method="BF",match_threshold=10):
        self.image_list = image_list
        self.keypoints = []
        self.descriptors = []
        self.matches = []
        self.k = k
        self.method = method
        self.ratio = 0.7
        self.match_threshold = match_threshold

    def extractSIFTFeatures(self,img):
        # Extract SIFT features from an image
        # Input:
        #   img: image
        # Output:
        #   features: SIFT features
        #   descriptors: SIFT descriptors

        # Create SIFT object
        sift = cv.SIFT_create()
        # Find keypoints and descriptors
        keypoints, descriptors = sift.detectAndCompute(img,None)
        return keypoints, descriptors
    
    def computeFeatures(self):
        # Compute features for all images
        # Input:
        #   self.image_list: list of images
        # Output:
        #   self.keypoints: list of keypoints for each image
        #   self.descriptors: list of descriptors for each image

        # Loop through all images
        for i in range(len(self.image_list)):
            # Extract SIFT features
            keypoints, descriptors = self.extractSIFTFeatures(self.image_list[i])
            # Add keypoints and descriptors to lists
            self.keypoints.append(keypoints)
            self.descriptors.append(descriptors)

    def matchFeatures(self,descriptors_1,descriptors_2):
        # Match SIFT features from two images
        # Input:
        #   descriptors_1: SIFT descriptors from image 1
        #   descriptors_2: SIFT descriptors from image 2
        #   method: matching method, "BF" for brute force, "FLANN" for FLANN
        #   k: number of nearest neighbors in KNN
        # Output:
        #   matches: SIFT matches

        if self.method == "BF":
            # Create BFMatcher object
            bf = cv.BFMatcher()
            # Match descriptors
            matches = bf.knnMatch(descriptors_1,descriptors_2, k=self.k)
        elif self.method == "FLANN":
            FLANN_INDEX_KDTREE = 1
            index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
            search_params = dict(checks = 50)
            flann = cv.FlannBasedMatcher(index_params, search_params)
            matches = flann.knnMatch(descriptors_1,descriptors_2,k=self.k)

        good_matches = []
        for m,n in matches:
            if m.distance < self.ratio*n.distance:
                good_matches.append(m)
        return good_matches
    
    def computeMatches(self):
        # Compute matches for all images
        # Input:
        #   self.image_list: list of images
        #   self.descriptors: list of descriptors for each image
        # Output:
        #   matches: list of matches for each image pair

        # Loop through all image pairs
        for i in range(len(self.image_list)):
            for j in range(i+1,len(self.image_list)):
                # Match features
                matches_ij = self.matchFeatures(self.descriptors[i],self.descriptors[j])
                # Add matches to list
                self.matches.append(matches_ij)
                

    def plot_matches(self, image_index_1, image_index_2):
        # Plot SIFT matches
        # Input:
        #   img_1: image 1
        #   img_2: image 2
        #   keypoints_1: SIFT keypoints from image 1
        #   keypoints_2: SIFT keypoints from image 2
        #   matches: SIFT matches
        #   title: title of the plot

        img_1 = self.image_list[image_index_1]
        img_2 = self.image_list[image_index_2]
        keypoints_1 = self.keypoints[image_index_1]
        keypoints_2 = self.keypoints[image_index_2]
        matches = self.matches[image_index_1*(len(self.image_list)-2)+image_index_2-1]
        title = "SIFT matches between image " + str(image_index_1) + " and image " + str(image_index_2)

        img_1 = cv.cvtColor(img_1, cv.COLOR_BGR2RGB)
        img_2 = cv.cvtColor(img_2, cv.COLOR_BGR2RGB)

        # Create a new plot
        plt.figure(figsize=(15, 15))
        plt.axis('off')
        # Set plot title
        plt.title(title)
        # Plot image 1
        plt.imshow(img_1)
        # Get image 1 dimensions
        h_1, w_1 = img_1.shape[:2]
        # Plot image 2
        plt.imshow(img_2, extent=(w_1, w_1 + img_2.shape[1], img_2.shape[0], 0))
        # Plot lines between matches
        for match in matches:
            # Get the matching keypoints for each of the images
            img1_idx = match.queryIdx
            img2_idx = match.trainIdx
            # x - columns
            # y - rows
            (x1,y1) = keypoints_1[img1_idx].pt
            (x2,y2) = keypoints_2[img2_idx].pt
            plt.plot([x1, x2+w_1], [y1, y2], 'c', linewidth=1)
            plt.plot(x1, y1, 'ro')
            plt.plot(x2+w_1, y2, 'go')
        # Show the plot
        plt.show()
    
    def computeAdjacencyMatrix(self):
        # Compute adjacency matrix from matches
        # Input:
        #   matches: list of matches for each image pair
        # Output:
        #   adjMatrix: adjacency matrix

        # Initialize adjacency matrix
        adjMatrix = np.zeros((len(self.image_list),len(self.image_list)))
        # Loop through all image pairs
        for i in range(len(self.image_list)):
            for j in range(i+1,len(self.image_list)):
                # If there are matches between image i and image j
                if len(self.matches[i*(len(self.image_list)-2)+j-1]) > self.match_threshold:
                    # Set adjacency matrix entries to 1
                    adjMatrix[i,j] = 1
                    adjMatrix[j,i] = 1
        return adjMatrix
    
    def run(self):
        self.computeFeatures()
        self.computeMatches()
        adjMatrix = self.computeAdjacencyMatrix()
        return adjMatrix


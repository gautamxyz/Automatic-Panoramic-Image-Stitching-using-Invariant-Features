import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
from itertools import compress
from logger import logger

class featureExtraction:
    def __init__(self,image_list,k=2,featureDetection="SIFT",featureMatching="BF",match_threshold=100,maximum_matches=6,feature_correspondences=4,RANSAC_iterations=500):
        self.image_list = self.initialize_images(image_list)
        self.num_images = len(image_list)
        self.keypoints = []
        self.descriptors = []
        self.matches = np.empty((self.num_images,self.num_images),dtype=object)
        self.inliers = np.zeros((self.num_images,self.num_images),dtype=int)
        self.homographies = self.initialize_homographies()
        self.adjacency_matrix = np.zeros((self.num_images,self.num_images),dtype=int)
        self.k = k
        self.featureMatching = featureMatching
        self.featureDetection = featureDetection
        self.ratio = 0.7
        self.match_threshold = match_threshold
        self.maximum_matches = maximum_matches
        self.feature_correspondences = feature_correspondences
        self.RANSAC_iterations = RANSAC_iterations


    def initialize_images(self,image_list):
        # Convert all images in image_list to grayscale
        # Input:
        #   image_list: list of images
        # Output:
        #   image_list: list of grayscale images
        for i in range(len(image_list)):
            image_list[i] = cv.cvtColor(image_list[i], cv.COLOR_BGR2GRAY)
        return image_list

    def initialize_homographies(self):
        # Initialize homographies to identity matrix
        # Input:
        #   self.num_images: number of images
        # Output:
        #   self.homographies: list of homographies
        homographies = np.empty((self.num_images,self.num_images),dtype=object)
        for i in range(self.num_images):
            for j in range(self.num_images):
                homographies[i,j] = None

        return homographies
    
    def process_maximum_matches(self):
        # Only consider the top maximum_matches matches for and image i
        # Input:
        #   self.matches: list of matches for each image pair
        #   self.maximum_matches: maximum number of matches to consider
        # Output:
        #   self.matches: list of matches for each image pair

        # Loop through all image pairs
        for i in range(self.num_images):
            match_count = np.zeros(self.num_images)
            for j in range(self.num_images):
                if i == j:
                    continue
                match_count[j] = len(self.matches[i,j])
            # Sort matches by number of matches
            sorted_matches = np.argsort(match_count)[::-1]
            # Select top maximum_matches matches
            if len(sorted_matches) > self.maximum_matches:
                sorted_matches = sorted_matches[:self.maximum_matches]

            # Loop through all images
            for j in range(self.num_images):
                if i == j:
                    continue
                if len(self.matches[i,j]) == 0:
                    self.matches[i,j] = None
                if j not in sorted_matches:
                    self.matches[i,j] = None

    def get_H_matrix(self,image_index_1,image_index_2):
        # Get homography matrix for image pair
        # Input:
        #   image_index_1: index of image 1
        #   image_index_2: index of image 2
        # Output:
        #   H: homography matrix

        matches = self.matches[image_index_1,image_index_2]
        keypoints_1 = self.keypoints[image_index_1]
        keypoints_2 = self.keypoints[image_index_2]

        src_pts = np.empty((len(matches),2),dtype=float)
        dst_pts = np.empty((len(matches),2),dtype=float)

        # Create correspondence matrix
        correspondences = np.zeros((len(matches),self.feature_correspondences),dtype=float)
        for i in range(len(matches)):
            m = matches[i]
            correspondences[i,0] = src_pts[i,0] = keypoints_1[m.queryIdx].pt[0]
            correspondences[i,1] = src_pts[i,1] = keypoints_1[m.queryIdx].pt[1]
            correspondences[i,2] = dst_pts[i,0] = keypoints_2[m.trainIdx].pt[0]
            correspondences[i,3] = dst_pts[i,1] = keypoints_2[m.trainIdx].pt[1]

        src_pts, dst_pts = np.float32(src_pts), np.float32(dst_pts)

        # Compute homography matrix
        H, mask = cv.findHomography(src_pts, dst_pts, cv.RANSAC, ransacReprojThreshold=1.0)
        # Normalize homography matrix
        H = H/H[2,2]
        return H, mask, correspondences


    def validate_homography(self,image_index_1,image_index_2,H,correspondences,mask):
        # Validate homography matrix
        # Input:
        #   image_index_1: index of image 1
        #   image_index_2: index of image 2
        #   H: homography matrix
        #   correspondences: correspondence matrix
        # Output:
        #   valid: boolean indicating if homography is valid
        img_1 = self.image_list[image_index_1]
        img_2 = self.image_list[image_index_2]

        if H is None:
            return False
        else:
            alpha = 8.0
            beta = 0.3

            mask_1 = np.ones_like(img_1, dtype=np.uint8)
            mask_2 = np.ones_like(img_1, dtype=np.uint8)
            mask_2 = cv.warpPerspective(np.ones_like(img_2, dtype=np.uint8), dst=mask_2, M=H, dsize=mask_1.shape[::-1])
            overlap = mask_1 * mask_2

            area = np.sum(mask)
            matchpoints_1 = correspondences[:, :2]
            overlapping_matches = matchpoints_1[overlap[matchpoints_1[:, 1].astype(np.int64), matchpoints_1[:, 0].astype(np.int64)] == 1]

            return area > (alpha + (beta * overlapping_matches.shape[0]))


    def compute_homograpies(self):
        # Compute homographies for all images
        # Input:
        #   self.image_list: list of images
        #   self.keypoints: list of keypoints for each image
        #   self.matches: list of matches for each image pair
        # Output:
        #   self.homographies: list of homographies

        # Loop through all image pairs
        for i in range(self.num_images):
            for j in range(self.num_images):
                # Skip if image pair already computed
                if i == j:
                    continue
                if self.homographies[i,j] is not None:
                    continue
                if self.matches[i,j] is None:
                    self.homographies[i,j] = None
                    self.homographies[j,i] = None
                    self.inliers[i,j] = 0
                    self.inliers[j,i] = 0
                    continue
                # Compute homography
                H , mask, correspondences = self.get_H_matrix(i,j)
                # Validate homography
                isValid = self.validate_homography(i,j,H,correspondences,mask)

                if isValid:
                    self.homographies[i,j] = H
                    self.homographies[j,i] = np.linalg.inv(H)
                    self.inliers[i,j] = np.sum(mask)
                    self.inliers[j,i] = np.sum(mask)
                    self.matches[i,j] = list(compress(self.matches[i,j],mask))
                else:
                    self.homographies[i,j] = None
                    self.homographies[j,i] = None
                    self.inliers[i,j] = 0
                    self.inliers[j,i] = 0
                    self.matches[i,j] = None



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
    
    def extractSURFFeatures(self,img):
        # Extract SURF features from an image
        # Input:
        #   img: image
        # Output:
        #   features: SURF features
        #   descriptors: SURF descriptors

        # Create SURF object
        surf = cv.xfeatures2d.SURF_create()
        # Find keypoints and descriptors
        keypoints, descriptors = surf.detectAndCompute(img,None)
        return keypoints, descriptors
    
    def extractORBFeatures(self,img):
        # Extract ORB features from an image
        # Input:
        #   img: image
        # Output:
        #   features: ORB features
        #   descriptors: ORB descriptors

        # Create ORB object
        orb = cv.ORB_create()
        # Find keypoints and descriptors
        keypoints, descriptors = orb.detectAndCompute(img,None)
        return keypoints, descriptors
    
    def computeFeatures(self):
        # Compute features for all images
        # Input:
        #   self.image_list: list of images
        # Output:
        #   self.keypoints: list of keypoints for each image
        #   self.descriptors: list of descriptors for each image

        # Loop through all images
        for image in self.image_list:
            # Extract SIFT features
            if self.featureDetection == "SIFT":
                keypoints, descriptors = self.extractSIFTFeatures(image)
            elif self.featureDetection == "SURF":
                keypoints, descriptors = self.extractSURFFeatures(image)
            elif self.featureDetection == "ORB":
                keypoints, descriptors = self.extractORBFeatures(image)
            # Add keypoints and descriptors to lists
            self.keypoints.append(keypoints)
            self.descriptors.append(descriptors)

    def matchFeatures(self,descriptors_1,descriptors_2):
        # Match SIFT features from two images
        # Input:
        #   descriptors_1: SIFT descriptors from image 1
        #   descriptors_2: SIFT descriptors from image 2
        #   featureMatching: matching method, "BF" for brute force, "FLANN" for FLANN
        #   k: number of nearest neighbors in KNN
        # Output:
        #   matches: SIFT matches

        if self.featureMatching == "BF":
            # Create BFMatcher object
            bf = cv.BFMatcher()
            # Match descriptors
            matches = bf.knnMatch(descriptors_1,descriptors_2, k=self.k)
        elif self.featureMatching == "FLANN":
            FLANN_INDEX_KDTREE = 1
            index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
            search_params = dict(checks = 50)
            flann = cv.FlannBasedMatcher(index_params, search_params)
            matches = flann.knnMatch(descriptors_1,descriptors_2,k=self.k)

        good_matches = []
        for m,n in matches:
            if m.distance < self.ratio*n.distance:
                good_matches.append(m)

        if len(good_matches) > self.match_threshold:
            return good_matches
        else:
            return []
    
    def computeMatches(self):
        # Compute matches for all images
        # Input:
        #   self.image_list: list of images
        #   self.descriptors: list of descriptors for each image
        # Output:
        #   matches: list of matches for each image pair

        # Loop through all image pairs
        for i in range(self.num_images):
            for j in range(self.num_images):
                # Skip if image pair already computed
                if i == j:
                    continue
                # Compute matches
                matches = self.matchFeatures(self.descriptors[i],self.descriptors[j])
                # Add matches to list
                self.matches[i,j] = matches
                

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
        matches = self.matches[image_index_1,image_index_2]
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

        # Loop through all image pairs
        for i in range(self.num_images):
            for j in range(self.num_images):
                # Skip if image pair already computed
                if i == j:
                    continue
                # Compute matches
                matches = self.matches[i,j]
                if matches is None:
                    continue
                # Add matches to list
                self.adjacency_matrix[i,j] = 1 if len(matches) > 0 else 0
    
    def run(self):
        logger.info("Extracting SIFT features...")
        self.computeFeatures()
        logger.info("Generating matches...")
        self.computeMatches()
        logger.info("Processing matches...")
        self.process_maximum_matches()
        logger.info("Computing homographies...")
        self.compute_homograpies()
        logger.info("Computing adjacency matrix...")
        self.computeAdjacencyMatrix()



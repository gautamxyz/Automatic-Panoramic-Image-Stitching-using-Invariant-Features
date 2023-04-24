import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt


class Stitcher():

    def __init__(self, images, homographies, gain_list):
        self.images = images
        self.gains = gain_list
        self.homographies = homographies
        self.offset = np.eye(3)
        self.panorama = None
        self.weights = None
        self.width = 0
        self.height = 0
        self.preProcessImages()

    def apply_filter(self,image,filter):
        # filter is a 1x3 array
        # image is a 2D array with 3 channels
        # returns a 2D array with 3 channels

        # convert image to float
        image = image.astype(np.float32)
        # apply filter
        image[:,:,0] = image[:,:,0]*filter[0]
        image[:,:,1] = image[:,:,1]*filter[1]
        image[:,:,2] = image[:,:,2]*filter[2]
        # convert back to uint8
        image = image.astype(np.uint8)
        return image
    
    def preProcessImages(self):
        for i in range(len(self.images)):
            self.images[i] = self.apply_filter(self.images[i],self.gains[i])
            self.images[i] = cv.cvtColor(self.images[i], cv.COLOR_BGR2RGB)

    def getWeightsArray(self,size):
        if size % 2 == 1:
            return np.concatenate(
                [np.linspace(0, 1, (size + 1) // 2),
                 np.linspace(1, 0, (size + 1) // 2)[1:]]
            )
        else:
            return np.concatenate([np.linspace(0, 1, size // 2), np.linspace(1, 0, size // 2)])

    def generateWeights(self,shape):
        h, w = shape[:2]
        h_weights = self.getWeightsArray(h)[: , np.newaxis]
        w_weights = self.getWeightsArray(w)[: , np.newaxis].T
        return h_weights @ w_weights

    # Transform a given list of points using the given homography matrix.
    def transformPoints(self, points, H):
        new_points = []
        for point in points:
            point = np.asarray([[point[0][0], point[1][0], 1]]).T
            new_point = H @ point
            new_points.append(new_point[0:2] / new_point[2])
        return np.array(new_points)

    # Get the bounding box of the current panaroma image using the corners of the images.
    def getBoundingBox(self, corners_images):
        top_right_x = np.max([corners_image[1][0]
                             for corners_image in corners_images])
        bottom_right_x = np.max([corners_images[3][0]
                                for corners_images in corners_images])

        bottom_left_y = np.max([corners_images[2][1]
                               for corners_images in corners_images])
        bottom_right_y = np.max([corners_images[3][1]
                                for corners_images in corners_images])

        width = int(np.ceil(max(bottom_right_x, top_right_x)))
        height = int(np.ceil(max(bottom_right_y, bottom_left_y)))

        width = min(width, 5000)
        height = min(height, 4000)
        return width, height

    # Generate the corners of the panaroma image.
    # This is used to determine offset and size of the panaroma image.
    def generateCorners(self, image, H):
        h, w = image.shape[:2]
        top_left = np.asarray([[0, 0]]).T
        top_right = np.asarray([[w, 0]]).T
        bottom_left = np.asarray([[0, h]]).T
        bottom_right = np.asarray([[w, h]]).T
        corners = [top_left, top_right, bottom_left, bottom_right]
        new_corners = self.transformPoints(corners, H)
        return new_corners

    # Generate the offset matrix required for adding the current image to the panaroma image.
    # Transforms the corners of the current image to the panaroma image frame.
    def generateOffset(self, corners):
        top_left, top_right, bottom_left, bottom_right = corners
        min_x = min(top_left[0], bottom_left[0])
        min_y = min(top_left[1], top_right[1])
        return np.array(
            [
                [1, 0, max(0, -float(min_x))],
                [0, 1, max(0, -float(min_y))],
                [0, 0, 1],
            ],
            dtype=np.float32,
        )
    def updatePanaroma(self, image, H):
        new_corners = self.generateCorners(image, H)
        required_offset = self.generateOffset(new_corners)
        shifted_corners_image = self.generateCorners(image, required_offset @ H)

        if self.panorama is None:
            self.width, self.height = self.getBoundingBox(
                [shifted_corners_image])
            # self.panorama = np.zeros((self.height, self.width, 3), dtype=np.uint8)
        else:
            current_corners = self.generateCorners(self.panorama, required_offset)
            self.width, self.height = self.getBoundingBox(
                [current_corners, shifted_corners_image])
            
        return required_offset

    def addImage(self, image, H):
        # Shift homography by current offset
        H = self.offset @ H
        # Generate new size and offset after adding the current image
        required_offset = self.updatePanaroma(image, H)
        # Warp the current image to the panaroma image frame with the new offset
        warped_image = cv.warpPerspective(
            image, required_offset @ H, (self.width, self.height))
        # Add the warped image to the panaroma image
        if self.panorama is None:
            self.panorama = np.zeros_like(warped_image)
            self.weights = np.zeros_like(warped_image)
        else:
            self.panorama = cv.warpPerspective(
                self.panorama, required_offset, (self.width, self.height))
            self.weights = cv.warpPerspective(
                self.weights, required_offset, (self.width, self.height))

        image_weights = self.generateWeights(image.shape)
        image_weights = np.repeat(
            cv.warpPerspective(image_weights, required_offset @ H, (self.width, self.height))[:, :, np.newaxis], 3, axis=2
        )

        normalized_weights = np.zeros_like(self.weights, dtype=np.float64)
        np.divide(
            self.weights, (self.weights+image_weights), where= self.weights+image_weights != 0, out=normalized_weights
        )

        self.panorama = np.where(
            np.logical_and(
                np.repeat(np.sum(self.panorama, axis=2)[:,:,np.newaxis], 3, axis=2) == 0,
                np.repeat(np.sum(warped_image, axis=2)[:,:,np.newaxis], 3, axis=2) == 0
            ),
            0,
            warped_image * (1 - normalized_weights) + self.panorama * normalized_weights,
        ).astype(np.uint8)

        max_weights = np.max(self.weights + image_weights)
        self.weights = (self.weights + image_weights) / max_weights
        # Update the offset matrix
        self.offset = required_offset @ self.offset

    def add_weights(self,weights_matrix ,image,idx):
        H = self.offset @ self.homographies[idx]
        added_offset = self.updatePanaroma(image,H)
        weights = self.generateWeights(image.shape)
        size = (self.width, self.height)
        weights = cv.warpPerspective(weights, added_offset @ H, size)[:, :, np.newaxis]

        if weights_matrix is None:
            weights_matrix = weights
        else:
            weights_matrix = cv.warpPerspective(weights_matrix, added_offset, size)

            if len(weights_matrix.shape) == 2:
                weights_matrix = weights_matrix[:, :, np.newaxis]

            weights_matrix = np.concatenate([weights_matrix, weights], axis=2)

        self.offset = added_offset @ self.offset
        return weights_matrix


    def getMaxWeightsMatrix(self):
        weights_matrix = None

        for idx,image in enumerate(self.images):
            weights_matrix = self.add_weights(weights_matrix,image,idx)

        weights_maxes = np.max(weights_matrix, axis=2)[:, :, np.newaxis]
        max_weights_matrix = np.where(
            np.logical_and(weights_matrix == weights_maxes, weights_matrix > 0), 1.0, 0.0
        )

        max_weights_matrix = np.transpose(max_weights_matrix, (2, 0, 1))

        return max_weights_matrix

    def get_cropped_weights(self,weights):
        cropped_weights = []
        for i, image in enumerate(self.images):
            cropped_weights.append(
                cv.warpPerspective(
                    weights[i], np.linalg.inv(self.offset @ self.homographies[i]), image.shape[:2][::-1]
                )
            )

        return cropped_weights

    def build_band_panorama(self,k,bands,size):
        # size = (self.height, self.width)
        pano_weights = np.zeros(size)
        pano_bands = np.zeros((*size, 3))
        weights = self.weights[k]

        for i, image in enumerate(self.images):
            weights_at_scale = cv.warpPerspective(weights[i], self.offset @ self.homographies[i], size[::-1])
            pano_weights += weights_at_scale
            pano_bands += weights_at_scale[:, :, np.newaxis] * cv.warpPerspective(
                bands[i], self.offset @ self.homographies[i], size[::-1]
            )
        return np.divide(
            pano_bands, pano_weights[:, :, np.newaxis], where=pano_weights[:, :, np.newaxis] != 0
        )

    def multiBandBlending(self,numBands,sigma):
        maxWeightsMatrix = self.getMaxWeightsMatrix()
        size = maxWeightsMatrix.shape[1:]
        maxWeights = self.get_cropped_weights(maxWeightsMatrix)

        self.weights = [[cv.GaussianBlur(maxWeights[i],(0,0),2*sigma) for i in range(len(self.images))]]
        sigmaImages = [cv.GaussianBlur(self.images[i],(0,0),sigma) for i in range(len(self.images))]

        bands = [
            [
                np.where(
                    self.images[i].astype(np.int64) - sigmaImages[i].astype(np.int64) > 0,
                    self.images[i] - sigmaImages[i],
                    0,
                )
                for i in range(len(self.images))
            ]
        ]

        for k in range(1, numBands - 1):
            sigma_k = np.sqrt(2 * k + 1) * sigma
            self.weights.append(
                [cv.GaussianBlur(self.weights[-1][i], (0, 0), sigma_k) for i in range(len(self.images))]
            )

            oldSigmaImages = sigmaImages

            sigmaImages = [
                cv.GaussianBlur(old_sigma_image, (0, 0), sigma_k)
                for old_sigma_image in oldSigmaImages
            ]
            bands.append(
                [
                    np.where(
                        oldSigmaImages[i].astype(np.int64) - sigmaImages[i].astype(np.int64) > 0,
                        oldSigmaImages[i] - sigmaImages[i],
                        0,
                    )
                    for i in range(len(self.images))
                ]
            )

        self.weights.append([cv.GaussianBlur(self.weights[-1][i], (0, 0), sigma_k) for i in range(len(self.images))])
        bands.append([sigmaImages[i] for i in range(len(self.images))])

        self.panorama = np.zeros((*maxWeightsMatrix.shape[1:], 3), dtype = np.uint8)
        
        for k in range(0, numBands):
            # plt.imshow(self.panorama)
            # plt.show()
            temp = self.build_band_panorama(k,bands[k],size).astype(np.uint8)
            self.panorama += temp
            self.panorama[self.panorama < 0] = 0
            self.panorama[self.panorama > 255] = 255

    def stitch(self):
        for idx in range(len(self.images)):
            image = self.images[idx]
            H = self.homographies[idx]
            self.addImage(image, H)

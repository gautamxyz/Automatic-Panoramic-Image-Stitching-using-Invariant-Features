import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt


class Stitcher():

    def __init__(self, images, homographies):
        self.images = images
        self.homographies = homographies
        self.offset = np.eye(3)
        self.panaroma = None
        self.weights = None
        self.width = 0
        self.height = 0

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

        # width = min(width, 5000)
        # height = min(height, 4000)
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

        if self.panaroma is None:
            self.width, self.height = self.getBoundingBox(
                [shifted_corners_image])
            # self.panaroma = np.zeros((self.height, self.width, 3), dtype=np.uint8)
        else:
            current_corners = self.generateCorners(self.panaroma, required_offset)
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
        if self.panaroma is None:
            self.panaroma = np.zeros_like(warped_image)
            self.weights = np.zeros_like(warped_image)
        else:
            self.panaroma = cv.warpPerspective(
                self.panaroma, required_offset, (self.width, self.height))
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

        self.panaroma = np.where(
            np.logical_and(
                np.repeat(np.sum(self.panaroma, axis=2)[:,:,np.newaxis], 3, axis=2) == 0,
                np.repeat(np.sum(warped_image, axis=2)[:,:,np.newaxis], 3, axis=2) != 0
            ),
            0,
            warped_image * (1 - normalized_weights) + self.panaroma * normalized_weights,
        ).astype(np.uint8)

        plt.imshow(image)
        plt.show()

        max_weights = np.max(self.weights + image_weights)
        self.weights = (self.weights + image_weights) / max_weights


    def stitch(self):
        for idx in range(len(self.images)):
            image = self.images[idx]
            H = self.homographies[idx]
            self.addImage(image, H)
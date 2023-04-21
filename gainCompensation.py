import numpy as np
import cv2
from featureExtraction import featureExtraction

def setOverlap(img1_idx,img2_idx,homographies,imgList):
    """
    Compute and set the overlap region between the two images.
    """
    mask_a = np.ones_like(imgList[img1_idx],dtype=np.uint8)
    mask_b = cv2.warpPerspective(np.ones_like(imgList[img2_idx],dtype=np.uint8),homographies[img1_idx][img2_idx],mask_a.shape[::-1])
    overlap = mask_a * mask_b
    area_overlap = overlap.sum()
    return overlap,area_overlap

def setIntensities(img1_idx,img2_idx,homographies,imgList,overlap):
    """
    Compute the intensities of the two images in the overlap region.
    """
    inverse_overlap = cv2.warpPerspective(overlap,np.linalg.inv(homographies[img1_idx][img2_idx]),imgList[img2_idx].shape[1::-1])
    Iab = (
        np.sum(
            imgList[img1_idx] * np.repeat(overlap[:, :, np.newaxis], 3, axis=2),
            axis=(0, 1),
        )
        / overlap.sum()
    )
    Iba = (
        np.sum(
            imgList[img2_idx] * np.repeat(inverse_overlap[:, :, np.newaxis], 3, axis=2),
            axis=(0, 1),
        )
        / inverse_overlap.sum()
    )
    return Iab,Iba



def gainCompensation(imgList,matches,homographies,sigma_n: float = 10.0, sigma_g: float = 0.1,gainList=[]):
    """
    Compute the gain compensation for each image, and save it into the gain Compensation Array.
    Parameters
    ----------
    images : Images of the panorama.
    sigma_n : float, optional
        Standard deviation of the normalized intensity error, by default 10.0
    sigma_g : float, optional
        Standard deviation of the gain, by default 0.1
    """

    coefficients = []
    results = []

    for idx, img in enumerate(imgList):
        coefs = [np.zeros(3) for _ in range(len(imgList))]
        result = np.zeros(3)

        # travel in the k'th row and k'th column of the matrix
        for i in range(len(matches)):
            if i!=idx:
                if(matches[i][k]!=None):
                    overlap,area_overlap = setOverlap(i,k,homographies,imgList)
                    Iab,Iba = setIntensities(i,k,homographies,imgList,overlap)
                    coefs[k] += area_overlap * (
                        (2*Iab ** 2 / sigma_n ** 2) + (1 / sigma_g ** 2)
                    )
                    coefs[i] -= (
                        (2 / sigma_n ** 2) * area_overlap * Iab * Iba
                    )
                    result += area_overlap / sigma_g ** 2
                if(matches[k][i]!=None):
                    overlap,area_overlap = setOverlap(k,i,homographies,imgList)
                    Iab,Iba = setIntensities(k,i,homographies,imgList,overlap)
                    coefs[k] += area_overlap * (
                        (2*Iab ** 2 / sigma_n ** 2) + (1 / sigma_g ** 2)
                    )
                    coefs[i] -= (
                        (2 / sigma_n ** 2) * area_overlap * Iab * Iba
                    )
                    result += area_overlap / sigma_g ** 2

                coefficients.append(coefs)
                results.append(result)

    coefficients = np.array(coefficients)
    results = np.array(results)

    gains = np.zeros_like(results)

    for channel in range(coefficients.shape[2]):
        coefs = coefficients[:, :, channel]
        res = results[:, channel]
        gains[:, channel] = np.linalg.solve(coefs, res)

    max_pixel_value = np.max([img for img in imgList])

    if gains.max() * max_pixel_value > 255:
        gains = gains / (gains.max() * max_pixel_value) * 255

    for i, img in enumerate(imgList):
        gainList.append(gains[i])



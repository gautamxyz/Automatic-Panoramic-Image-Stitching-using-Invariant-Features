import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import os
import networkx as nx
from skimage.metrics import structural_similarity as ssim
from featureExtraction import featureExtraction
from bundleAdjustment import bundleAdjustment
from gainCompensation import gainCompensation
from stitcher import Stitcher

def dataLoader(path,limit=None):
    img_list = []
    for file in os.listdir(path):
        img = cv.imread(os.path.join(path,file))
        img_list.append(img)
        if limit!=None and len(img_list)==limit:
            break
    return img_list


def generatePanaroma(img_list,args):

    featureDetection = args["featureDetection"] or "SIFT"
    featureMatching = args["featureMatching"] or "FLANN"
    startPoint = args["startPoint"] or "degree"
    gain = args["gain"] or False
    blending = args["blending"] or "MultiBand"
    numBands = args["numBands"] or 7
    sigma = args["sigma"] or 1
    opencv = args["opencv"] or False

    fe = featureExtraction(img_list.copy(),featureDetection=featureDetection,featureMatching=featureMatching)
    fe.run()
    numMatches=np.zeros((len(img_list),len(img_list)))
    for i in range(len(img_list)):
        for j in range(len(img_list)):
            if fe.matches[i][j]==None:
                numMatches[i,j]=0
                continue
            numMatches[i,j]=-len(fe.matches[i][j]) # we take -ve wts as we want to retain the edges with more matches while applying MST
    BA=bundleAdjustment(fe.matches.copy(),numMatches,fe.homographies.copy(),img_list.copy(),fe.keypoints,startPoint=startPoint)
    BA.run()
    gain_list = []
    gainCompensation(img_list.copy(),fe.matches.copy(),fe.homographies.copy(),gainList=gain_list)

    panos = []
    for i in range(len(BA.paths)):
        path = BA.paths[i]
        img_list = []
        H_list = [np.eye(3)]
        for j in path:
            img_list.append(BA.imgList[j])
        for H in BA.bundleHomo[i]:
            H_list.append(H)

        if opencv:
            stitcher = cv.Stitcher.create()
            (status, pano) = stitcher.stitch(img_list)
            pano = cv.cvtColor(pano, cv.COLOR_BGR2RGB)
            panos.append(pano)
            continue

        st = Stitcher(img_list.copy(),H_list,gain_list,setGain=gain,blending=blending,numBands=numBands,sigma=sigma)
        st.stitch()
        pano = st.panorama
        panos.append(pano)
    return panos
    
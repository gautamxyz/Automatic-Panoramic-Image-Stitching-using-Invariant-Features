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
    file_list = os.listdir(path)
    for file in file_list:
        if os.path.isfile(os.path.join(path,file))==False:
            continue
        img = cv.imread(os.path.join(path,file))
        img_list.append(img)
        if limit!=None and len(img_list)==limit:
            break
    return img_list

def get_network(adjacency_matrix):
    G = nx.Graph(adjacency_matrix)
    return G

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
    G = get_network(fe.adjacency_matrix)
    numMatches=np.zeros((len(img_list),len(img_list)))
    for i in range(len(img_list)):
        for j in range(len(img_list)):
            if fe.matches[i][j]==None:
                numMatches[i,j]=0
                continue
            numMatches[i,j]=-len(fe.matches[i][j]) # we take -ve wts as we want to retain the edges with more matches while applying MST
    BA=bundleAdjustment(fe.matches.copy(),numMatches,fe.homographies.copy(),img_list.copy(),fe.keypoints,startPoint=startPoint)
    BA.run()
    color_map = []
    for node in G:
        if node not in BA.srcs:
            color_map.append('#00b4d9')
        else:
            color_map.append('green')

    gain_list = []
    if gain:
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
    return color_map,G,panos
    
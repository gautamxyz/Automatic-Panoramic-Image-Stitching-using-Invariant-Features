import cv2 as cv
import numpy as np
import networkx as nx
import scipy.sparse.csgraph as csgraph
import scipy.optimize
from logger import logger

# define the residual function
def get_diff(v1,v2):
        diff_v = []
        n = v1.shape[0]
        for i in range(n):
            x1 = v1[i][0][0]
            y1 = v1[i][0][1]
            x2 = v2[i][0][0]
            y2 = v2[i][0][1]
            val = np.sqrt((x1-x2)**2 + (y1-y2)**2)
            diff_v.append(h(val))
        return np.array(diff_v)
    
def h(x):
    sigma = 5
    if x < sigma:
        return x**2
    else:
        return 2*sigma*x - sigma**2



def residual(H_init, src_pts_list, dst_pts_list):
    error = []
    for src_pts, dst_pts in zip(src_pts_list, dst_pts_list):
        dst_pts_pred = cv.perspectiveTransform(dst_pts, H_init.reshape(3,3))
        # take L2 norm of the difference between the predicted and actual points
        diff_vector = get_diff(dst_pts_pred, src_pts)
        # convert to 1D array
        error.extend(diff_vector)
    error = np.array(error)
    # print(error.shape)
    return error
    # dst_pts_ = cv.perspectiveTransform(dst_pts, H.reshape(3,3))
    # return (dst_pts_ - src_pts).reshape(-1)



class bundleAdjustment():

    def __init__(self,matches, numMatches, homographies, imgList,keypoints, startPoint="degree"):
        self.matches = matches
        self.numMatches = numMatches
        self.homographies = homographies
        self.imgList = imgList
        self.kp = keypoints
        self.startPoint = startPoint
        self.path = []
        self.paths = []
        self.bundleHomo = []
        self.initialHomo = []
        self.parents = []
        self.srcs = []

    def bfs(self, G, node):

        # create a queue for bfs
        queue = []
        # mark all the nodes as not visited
        visited = [False] * len(self.imgList)
        # mark the source node as visited and enqueue it
        visited[node] = True
        queue.append(node)
        while queue:
            # dequeue a vertex from queue and print it
            s = queue.pop(0)
            self.path.append(s)
            # get adjacent vertices of the dequeued vertex s
            # if a adjacent has not been visited, then mark it visited and enqueue it
            for i in G.neighbors(s):
                if visited[i] == False:
                    visited[i] = True
                    queue.append(i)

    def buildNetwork(self,mode):
        G = nx.Graph(self.numMatches)
        # create connected components
        connected_components = list(nx.connected_components(G))
        # if a connected component has just 1 node, it is a single image, remove it
        connected_components = [x for x in connected_components if len(x) > 1]
        num_connected_components = len(connected_components)
        logger.info("Number of panoramas detected: "+ str(num_connected_components))
        # for each connected compnent apply mst algorithm
        for i in range(num_connected_components):
            # get the connected component
            connected_component = connected_components[i]
            # get the subgraph of the connected component
            subgraph = G.subgraph(connected_component)
            # get the minimum spanning tree of the subgraph
            # get weights of the edges from numMatches
            mst = nx.minimum_spanning_tree(subgraph)
            #  plot the mst
            # nx.draw(mst, with_labels=True, font_weight='bold')
            # plt.show()
            # find the node that has maximum number of matches with its neighbors
            # get the nodes of the mst
            nodes = list(mst.nodes())
            max_matches = 0
            max_degree = 0
            index_node = 0
            if mode == "degree":
                for j in range(len(nodes)):
                    # get the degree of the node
                    degree = mst.degree(nodes[j])
                    if degree > max_degree:
                        max_degree = degree
                        index_node = j
            elif mode == "matches":
                for j in range(len(nodes)):
                    # check matches between with the neighboring nodes
                    # get neighbors of the node
                    neighbors = list(mst.neighbors(nodes[j]))
                    # get the number of matches between the node and its neighbors
                    num_matches = 0
                    for neighbor in neighbors:
                        num_matches += self.numMatches[nodes[j]][neighbor]
                    if num_matches > max_matches:
                        max_matches = num_matches
                        index_node = j

            # apply dfs from the node with maximum number of matches
            # print(index_node, nodes[index_node])
            src = nodes[index_node]
            self.srcs.append(src)
            parents = nx.predecessor(mst, src)
            self.parents.append(parents)
            self.bfs(mst, nodes[index_node])
            self.paths.append(self.path)
            self.path = []

    def homography_to_src(self,cur,parent):
        H = np.eye(3)
        while len(parent[cur]) != 0:
            par = parent[cur][0]
            if self.homographies[cur][par] is None:
                logger.info("No homography between {} and {}".format(cur,par))
            else: 
                H = np.matmul(self.homographies[cur][par],H)
            cur = par
        return H


    def bundleAdjuster(self, path, parent):
        # start with the first image, its homography is identity
        H_init = np.eye(3)
        # for every match between img[path[i]] and img[path[i+1]], compute residual
        # and update homography
        ba=[path[0]]
        ordered_H = []
        initial_H = []
        for i in range(1,len(path)):
            # get the matches between img[path[i]] and img[path[i+1]]
            src_pts_list = []
            dst_pts_list = []

            # estimate homography
            

            
            for j in ba:
                matches = self.matches[j][path[i]]
                if matches is None:
                    # print("No matches between ", j, " and ", path[i])
                    continue
                # get the homography between img[path[i]] and img[path[i+1]]
                # get the coordinates of the matched points
                # get the coordinates of the matched points in img[path[i]]
                src_pts = np.float32([self.kp[j][m.queryIdx].pt for m in matches]).reshape(-1,1,2)
                H_temp = self.homography_to_src(j,parent)
                src_pts_transformed = cv.perspectiveTransform(src_pts, H_temp)
                src_pts_list.append(src_pts_transformed)
                # get the coordinates of the matched points in img[path[i+1]]
                dst_pts = np.float32([self.kp[path[i]][m.trainIdx].pt for m in matches]).reshape(-1,1,2)
                dst_pts_list.append(dst_pts)
                # optimize the residual function using the Levenberg-Marquardt algorithm

            
            H_init = self.homography_to_src(path[i],parent)
            initial_H.append(H_init.reshape(3,3))
            H_init = H_init.flatten()
            result = scipy.optimize.least_squares(residual, H_init, args=(src_pts_list, dst_pts_list), method='lm')
            # update the homography
            H = result.x.reshape(3,3)
            # normalize the homography
            H = H/H[2,2]
            ordered_H.append(H)
            ba.append(path[i])

        return ordered_H, initial_H
            

    
    def run(self):
        logger.info("Building network and ordering...")
        self.buildNetwork(self.startPoint)
        # return
        logger.info("Performing bundle adjustment for each panorama...")
        for i in range(len(self.paths)):
            logger.info("Panorama "+str(i+1))
            ordered_H, initial_H = self.bundleAdjuster(self.paths[i], self.parents[i])
            self.bundleHomo.append(ordered_H)
            self.initialHomo.append(initial_H)
        logger.info("Completed bundle adjustment for all panoramas")
    

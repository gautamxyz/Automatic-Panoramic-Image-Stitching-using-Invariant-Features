import numpy as np
# Helper function for findConnectedComponents
def findConnectedComponentsHelper(adjMatrix, node, visited, components, numComponents):
    # Mark node as visited
    visited[node] = 1

    # Add node to the current component
    components[numComponents].append(node)

    # Loop through all nodes
    for i in range(adjMatrix.shape[0]):
        # If node i is connected to node and has not been visited
        if adjMatrix[node, i] == 1 and visited[i] == 0:
            # Find all nodes connected to node i
            findConnectedComponentsHelper(adjMatrix, i, visited, components, numComponents)

def findConnectedComponents(adjMatrix):
    # Find connected components in an adjacency matrix
    # Input:
    #   adjMatrix: adjacency matrix
    # Output:
    #   components: list of components, each component is a list of node indices
    #   numComponents: number of components

    # Initialize components
    components = []
    numComponents = 0

    # Initialize visited
    visited = np.zeros(adjMatrix.shape[0])

    # Loop through all nodes
    for i in range(adjMatrix.shape[0]):
        # If node i has not been visited
        if visited[i] == 0:
            # Initialize a new component
            components.append([])
            # Find all nodes connected to node i
            findConnectedComponentsHelper(adjMatrix, i, visited, components, numComponents)
            # Increment number of components
            numComponents += 1

    return components, numComponents

# use the function
adjMatrix = np.array(
[[1, 1, 1, 0, 0],
[1, 1, 1, 0, 0],
[1, 1, 1, 0, 0],
[0, 0, 0, 1, 1],
[0, 0, 0, 1, 1]])
components, numComponents = findConnectedComponents(adjMatrix)
print(components)
print(numComponents)

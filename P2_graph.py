import argparse
import numpy as np
import os.path as path
import matplotlib.pyplot as plt
import matplotlib.animation as animation

KREP = 1e-2
KATR = 1e-2


class Simulation:
    def __init__(self, graph, max_iterations):
        """Object for running the simulation"""
        self.graph = graph
        self.max_iterations = max_iterations
        self.ani = None

        #The following arguments implements an iteration tracker. which starts counting from 0 
        self.current_iteration = 0

    def run(self, frame_num: int):
        """Function that runs and update the plot"""

        #Ensuring that the current iteration is not above our threshold
        if self.current_iteration < self.max_iterations:
            #if true, then we apply the update function to calculate & apply the forces 
            self.graph.update()

            #counting this iteration to keep track of the total iterations
            self.current_iteration += 1

            #Update the plot
            self.plot(frame_num)

        #but maybe, we have already iterated over the program too many times (current iterations are above our threshold)
        else:
            #stopping the animation
            print(f"Simulation stopped after {self.max_iterations} iterations.")
            self.ani.event_source.stop()
        

    def line(self, edge):
        """Returns the points between two nodes"""
        return [edge.node1.x, edge.node2.x], [
            edge.node1.y,
            edge.node2.y,
        ]

    def init_plot(self):
        """Initialises the plot"""
        fig = plt.figure()
        self.ax = fig.add_subplot(1, 1, 1)
        return fig

    def plot(self, frame_num):
        """Plots the nodes and edges"""
        self.ax.clear()
        for edge in self.graph.edges:
            x, y = self.line(edge)
            self.ax.plot(x, y, "k", marker="o", markersize=10)
        self.ax.set_title(f"Frame num: {frame_num}")


def theta(x0, x1, y0, y1):
    return np.arctan2(y1 - y0, x1 - x0)


class Node:
    def __init__(self):
        """Node object, carries the location and forces"""
        self.x = 0.0  # Do not rename
        self.y = 0.0  # Do not rename
        self.fx = 0.0  # Do not rename
        self.fy = 0.0  # Do not rename

    def add_force(self, fx, fy):
        """Adding a force to the node's current force."""
        self.fx += fx
        self.fy += fy

    def repel(self, other: "Node"):
        """
        Calculates and applys the repulsive force between this node and another. 
        The repulsive force is computed based on the distance between the nodes, which is calculated from their positions. 
        The force follows an inverse relationship to the distance. 
        The calculated force is applied to both nodes. 

        Args:
        other (Node): The other node that is being repelled from this node. 

        Returns: 
        The force attributes (fx,fy) of both nodes.

        """
        #First we calculate the distance between the 2 nodes 
        dx = other.x - self.x
        dy = other.y - self.y

        #Calculating the sum of squared distances 
        distance_squared = dx**2 + dy**2

        #Checking if the distance is zero to avoid diving by zero
        if distance_squared == 0:
            #Nodes are in the same position, no repulsion
            return 0.0, 0.0 
        
        #Calculating the repulsive force
        distance = np.sqrt(distance_squared)
        
        f_rep = KREP / distance

        #Analyzing the force into its x and y components 
        fx = f_rep * dx / distance  
        fy = f_rep * dy / distance 

        #Debugging step
        #print(f"Repel: Node1({self.x}, {self.y}) Node2({other.x}, {other.y}) -> Force({fx}, {fy}), Distance: {distance}")

        # Applying forces to current node & the other node
        # Following Newton's 3rd law: for each pair of nodes, the force is the same but acts in opposite directions
        self.add_force(-fx, -fy)
        other.add_force(fx, fy)

        #The reuturn statement is optional, but it is added to maintain clarity in the code (I am a beginner)
        return fx, fy    

class Edge:
    def __init__(self, node1: Node, node2: Node):
        """Edge is the connection between two nodes

        Args:
            node1 (Node): Node at first end of the edge
            node2 (Node): Node at other end of the edge
        """
        self.node1 = node1  # Do not rename
        self.node2 = node2  # Do not rename

    def attract(self):
        """
        Calculates and applies the attractive force between teh 2 nodes connected by this edge. 

        The attractive force between the 2 nodes connected by an edge (self.node1, self.node2) is computed based on the distance between the nodes.
        This is done following Hooke's Law. 
        The force pulls the nodes towards each other and is applied in opposite direction to each node.

        The method avoids division by zero if the nodes are found in the exact same position.

        Args: 
        self (Edge): The Edge object which connects 2 nodes (node1 and node2) whose attractive forces will be calculated.

        Returns: 
        The force attributes (fx,fy) of both nodes.

        """
        #Calculating the distance between the 2 nodes
        dx = self.node2.x - self.node1.x
        dy = self.node2.y - self.node1.y

        distance_squared = dx**2 + dy**2

        #Checking if the distance is zero to avoid diving by zero
        if distance_squared == 0:
            #If true, nodes are in the same position, no repulsion
            return 0.0, 0.0 

        distance = np.sqrt(distance_squared)

        #Applying Hooke's Law for Attractive Force
        f_attr = KATR * distance_squared

        #Breaking the force into x anx y components
        fx = f_attr * dx / distance  
        fy = f_attr * dy / distance

        #Applying forces to both nodes 
        self.node1.add_force(fx, fy)
        self.node2.add_force(-fx, -fy)


class Graph:
    def __init__(self, nodes: int, edges: list[tuple[int, int]]):
        """Graph object, contains all the Nodes and Edges

        Args:
            nodes (int): Number of nodes to create
            edges (list[tuple[int, int]]): list of edge pairs (node index 1, node index 2)
        """

        self.nodes = []  # do not rename
        self.edges = []  # do not rename

        # Adds nodes number of Nodes
        for _ in range(nodes):
            self.nodes.append(Node())

        # Create all the Edges
        for edge in edges:
            self.edges.append(Edge(self.nodes[edge[0]], self.nodes[edge[1]]))

    @staticmethod
    def from_file(filename):
        """
        Opens the file through the path given in the command line. 
        Reads the first line of the file as the total number of nodes. 
        Reads every subsequent line as an edge connecting 2 nodes 
        """
        #Opening the data file 
        with open(filename, mode='r') as file: 

            #Reading number of nodes
            #The number of unique nodes is given in the first line of each data file
            nodes = int(file.readline().strip())

            #Reading number of edges by reading all subsequent lines after the first line
            #initializing an empty list to save the edges in it
            edges = []
            #looping over all lines
            for line in file:
                #The map function applies the integer to both numbers in one line
                node1, node2 = map(int, line.strip().split())
                edges.append((node1, node2))
        
            return Graph(nodes, edges)

    def update(self):
        """Calculates and applies all forces"""
        #First, we reset the forces on all nodes 
        for node in self.nodes: 
            node.fx = 0.0
            node.fy = 0.0

        #Calculating the repulsive forces for all node pairs
        #We use indexing to avoid calculating forces from one node to iteself but also to get unique node pairs
        #Unique node pairs mean that the pair node1, node2 and node2, node1 is the same and the force for this is calculated only once
        num_nodes = len(self.nodes)
        for i in range (num_nodes):
            for j in range (i + 1, num_nodes):
                node1 = self.nodes[i]
                node2 = self.nodes[j]

                #Calculating the repulsive force between nodes 1 and 2
                node1.repel(node2)
        
        #Calculating the attractive forces for all edges
        for edge in self.edges:
            edge.attract()
        
        #Updating the node positions based on the forces 
        for node in self.nodes:
            #adding the force calculated earlier to the respective x and y attributes of each node
            node.x += node.fx
            node.y += node.fy

    def init_locations(self):
        #Placing the nodes in a unit circle arrangement to have stanadard starting point
        n = len(self.nodes)
        for k in range(n):
            angle = 2 * np.pi * k / n #calculating the angle for each node
            self.nodes[k].x = np.cos(angle)
            self.nodes[k].y = np.sin(angle)


MAX_ITERATIONS = 1000

def main():
    # -------------------------------------
    # Command line input parser
    # Takes following positional args
    # - filename (string)
    # - max_iterations (int)
    # -------------------------------------
    arg_parser = argparse.ArgumentParser()

    #Adding argument to read filename from the commandline
    arg_parser.add_argument(
        "filename", 
        type = str, 
        help = "Requires the path to the file you wish to create a graph for"
    )

    #Adding argument to enter the number of iterations from the commandline
    arg_parser.add_argument(
        "max_iterations", 
        type = int,  
        help = "Will run the program for as many times as you define in this argument. If not specified, the default is 100. ", 
        default = 100
    )
    # -------------------------------------
    args = arg_parser.parse_args()

    if args.max_iterations < 1 or args.max_iterations > MAX_ITERATIONS:
        raise ValueError(f"Iterations must be between 1 and {MAX_ITERATIONS}")
    # -------------------------------------

    # Create graph
    graph = Graph.from_file(path.join("data", args.filename))

    # Set initial circular locations
    graph.init_locations()

    # Initialise the plot
    sim = Simulation(graph, args.max_iterations)
    fig = sim.init_plot()

    # And the run the animation
    sim.ani = animation.FuncAnimation(fig, sim.run, interval=5)
    # sim.ani.save('animation.gif', writer='imagemagick', fps=30)
    plt.show()


if __name__ == "__main__":
    main()

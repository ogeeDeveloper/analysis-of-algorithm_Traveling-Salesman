import timeit
# from Searches.aSearch import a_star_algorithm
# from Searches.bestFirstSearch import best_first_search
import heapq
from math import inf
from collections import OrderedDict

#Overview
# 1 should integrate all 14 parishes 
# 2 empirical compares these search methods

# code should have the following searches:
# 1 ●	A* Search  Method  
#2  ●	Best First Search Method
#3 ●	Greedy Search Method  

# class to create Vertex
class Vertex:
    """vertex with key and optional data payload"""

    def __init__(self, key, index, payload=None):
        self.key = key
        self.payload = payload  # data associated with Vertex
        self.index = index

    def __str__(self):
        return self.key

    def __repr__(self):
        return str(self)

#create graph
class Graph:
    def __init__(self, directed=True, max_vertexes=100):
        # initialize a dictionary to hold the graph
        self.matrix = [[None] * max_vertexes for _ in range(max_vertexes)]  # 2d array (list of lists)
        self.num_vertexes = 0  # current number of vertexes
        self.vertexes = {}  # vertex dict
        self.i_to_key = []  # list of keys to look up from index


    # Function to add vertex
    def add_vertex(self, key, payload=None):
        """ add vertex named key if it is not already in graph"""
        assert self.num_vertexes < len(self.matrix), "max vertexes reached,  can not add more!"
        
        if key not in self.vertexes:
            self.i_to_key.append(key)
            item_count = self.num_vertexes
            self.vertexes[key] = Vertex(key, item_count, payload)
            self.num_vertexes = item_count + 1

    def add_edge(self, from_key, to_key, weight=None):
        """ create vertexes if needed and then set weight into matrix"""
        self.add_vertex(from_key)
        self.add_vertex(to_key)
        self.matrix[self.vertexes[from_key].index][self.vertexes[to_key].index] = weight
    
    def get_vertex(self, key):
        return self.vertexes[key]

    def get_vertices(self):
        """returns the list of all vertices in the graph."""
        return self.vertexes.values()

    def __contains__(self, key):
        return key in self.vertexes

    def edges(self, from_key):
        """ return list of tuples (to_vertex, weight) of all edges from vertex_key key"""
        to_dim = self.matrix[self.vertexes[from_key].index]
        return [(g.vertexes[g.i_to_key[i]], w) for i, w in enumerate(to_dim) if w]

    def set_heuristics(self, heuristics={}):
        self.heuristics = heuristics

    def __str__(self):
        return str(self.__dict__)

    # returns all neighbors of given node
    def neighbors(self, node):
        try: return self.edges[node]
        except KeyError: return []

    # def cost(self, node1, node2):
    #     try: return self.edges[node1][node2]
    #     except: return inf

    # return cost of edge (from_node, to_node)
    def cost(self, from_node, to_node):
        from_point = from_node[0]
        to_point = to_node[0]
        return rectilinear(from_point, to_point)

    def best_first_serach(self):
        pass
        

    def greedy_search(self):
        pass

# wrapper for heapq
class PriorityQueue:
    def __init__(self):
        self.elements = []
    
    def empty(self):
        return len(self.elements) == 0
    
    def put(self, item, priority):
        heapq.heappush(self.elements, (priority, item))
    
    def get(self):
        return heapq.heappop(self.elements)[1]

def a_star_search(graph, start, goal, heuristic):
    frontier = PriorityQueue()
    frontier.put(start, 0)
    came_from = OrderedDict()
    cost_so_far = OrderedDict()
    came_from[start] = None
    cost_so_far[start] = 0
    
    iter = 0

    while not frontier.empty():
        current = frontier.get()
        
        if current[0] == goal and iter != 0:
            break
        
        # add all unexplored neighbors of current node to priority queue
        for next in graph.neighbors(current):
            # print("CURRENT:", current)
            # print("NEXT:", next, end='\n\n')
            new_cost = cost_so_far[current] + graph.cost(current, next)
            # update cost of next neighbor if applicable
            if next not in cost_so_far or new_cost < cost_so_far[next]:
                cost_so_far[next] = new_cost
                # nodes with lower from_cost + heuristic_cost have higher priority (lower number)
                priority = new_cost + heuristic(graph, next)
                frontier.put(next, priority)
                came_from[next] = current
        
        # since the keys for the start and end states are the same, add an iteration tracker to make sure sure algo doesn't immediate return
        iter += 1
    
    return came_from, cost_so_far

if __name__ == '__main__':
    # Create the graph
    g = Graph()

    # Connect Vertex(Node) using Edges
    g.add_edge('Hanover', 'St.James', 31.14)
    g.add_edge('Hanover', 'Westmoreland', 24.94)
    g.add_edge('St.Elizabeth', 'St.James', 49.03)
    g.add_edge('St.Elizabeth', 'Westmoreland', 38.86)
    g.add_edge('St.Elizabeth', 'Manchester',38.48)
    g.add_edge('St.James', 'Hanover', 31.14)
    g.add_edge('St.James', 'St.Elizabeth', 49.03)
    g.add_edge('St.James', 'Trelawny', 22.45)
    g.add_edge('St.James', 'Manchester', 64.20)
    g.add_edge('Trelawny', 'St.James', 22.45)
    g.add_edge('Trelawny', 'Manchester', 53.93)
    g.add_edge('Trelawny', 'St.Ann', 48.88)
    g.add_edge('Westmoreland', 'Hanover', 24.94)
    g.add_edge('Westmoreland', 'St.Elizabeth', 38.86)
    g.add_edge('Clarendon', 'Manchester', 28.34)
    g.add_edge('Clarendon', 'St.Catherine', 39.88)
    g.add_edge('Manchester', 'St.Elizabeth', 38.48)
    g.add_edge('Manchester', 'St.James', 64.20)
    g.add_edge('Manchester', 'Trelawny', 53.93)
    g.add_edge('Manchester', 'Clarendon', 28.34)
    g.add_edge('St.Ann', 'Trelawny', 48.86)
    g.add_edge('St.Ann', 'St.Catherine', 51.86)
    g.add_edge('St.Ann', 'St.Mary', 34.04)
    g.add_edge('St.Ann', 'St.Andrew', 64.37)
    g.add_edge('St.Catherine', 'Clarendon', 29.88)
    g.add_edge('St.Catherine', 'St.Ann', 51.86)
    g.add_edge('St.Catherine', 'Kingston', 18.08)
    g.add_edge('St.Mary', 'St.Ann', 34.04)
    g.add_edge('St.Mary', 'Portland', 50.43)
    g.add_edge('St.Mary', 'St.Andrew', 41.79)
    g.add_edge('Kingston', 'St.Catherine', 18.08)
    g.add_edge('Kingston', 'St.Andrew', 3.60)
    g.add_edge('Kingston', 'St.Thomas', 47.35)
    g.add_edge('Portland', 'St.Mary', 50.43)
    g.add_edge('Portland', 'St.Thomas', 31.68)
    g.add_edge('St.Andrew', 'St.Ann', 64.37)
    g.add_edge('St.Andrew', 'St.Mary', 41.79)
    g.add_edge('St.Andrew', 'Kingston', 3.60)
    g.add_edge('St.Thomas', 'Kingston', 47.35)
    g.add_edge('St.Thomas', 'Portland', 31.68)

    # Make graph undirected, create symmetric connections
    #graph.make_undirected()


    print("Table of the Matrix from this graph:")
    print(" ", " ".join([v.key for v in g.get_vertices()]))
    for i in range(g.num_vertexes):
        row = map(lambda x: str(x) if x else '.', g.matrix[i][:g.num_vertexes])
        print(g.i_to_key[i], "  ".join(row))

    # print(print_adjacency_graph(g))

    """ TODO: Map through all the nodes and find all the edgess connected to that node"""
    #check all edges connected connected to a gived node
    # print('\ng.edges("Hanover")', g.edges("Hanover"))

    #checks if a node is in the graph
    # print('"Hanover" in g', "Hanover" in g)
    # print('"St.James" in g', "St.James" in g)
    # print('"St.James" in g', "St.James" in g)

    # Menu to show the users a list of search options to search for
    print('\n\t\t\tSelect a search below by using their numbers\
       \nEnter 1 for A* search\
       \nEnter 2 for Best first search\
       \nEnter 3 for greedy search')
    response =int(input("\nSelect an option from above: "))

    start = input("\nEnter a parish from the table as start node: ")
    destination = input("\nEnter a parish from the table as destination node: ")

    if response == 1:
        print("A * Search")
        # Create heuristics
        heuristics=g.set_heuristics({'Hanover':4.0,
                    'St.Elizabeth':14.0,
                    'St.James':14.0,
                    'Trelawny':10.0,
                    'Westmoreland':16.0,
                    'Clarendon':18.0,
                    'Manchester':5.0,
                    'St.Ann':13.0,
                    'St.Catherine':6.0,
                    'St.Mary':19.0,
                    'Kingston':17.0,
                    'Portland':20.0,
                    'St.Andrew':3.0,
                    'St.Thomas':14.0})

        path = a_star_search(g, start, destination,heuristics)
        print(path)

    elif response == 2:
        print("Best first search")
        heuristic = g.set_heuristics({'Hanover':0,
                    'St.Elizabeth':0,
                    'St.James':0,
                    'Trelawny':0,
                    'Westmoreland':0,
                    'Clarendon':0,
                    'Manchester':0,
                    'St.Ann':0,
                    'St.Catherine':0,
                    'St.Mary':0,
                    'Kingston':0,
                    'Portland':0,
                    'St.Andrew':0,
                    'St.Thomas':0})

    elif response == 3:
        print("Greedy Search")
    
    else:
        print("\nEntered response was incorrect")

    #Does the Breath Best First Search

    #Record the starting time
    # startTime = timeit.default_timer()
    # shortestPath = a_star_algorithm(g, heuristics, start, destination)
class Node:
    def __init__(self, node_type, content = None, source = None, edges = None, id = None):
        self.id = id
        self.node_type = node_type
        self.source = source
        self.edges = edges or {}
        self.content = content
        self.degree = self.getDegree()

    def link(self,node, weight = 1):
        if node.id not in self.edges:
            self.edges[node.id] = weight
        else:
            self.edges[node.id] += weight
        self.degree = self.getDegree()

    def print(self):
        print(f"ID: {self.id}")
        print(f"Type: {self.node_type}")
        print(f"Source: {self.source}")
        print(f"Edges: {self.edges}")
        print(f"Content: {self.content}")
        
    def getDegree(self):
        return sum(self.edges.values())
        
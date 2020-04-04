"""This module contains a light weight graph representation."""

import copy
import os
from collections import OrderedDict

import networkx as nx


class Graph(object):
    """A light weight graph representation."""

    def __init__(self, graph):
        """Initialize a Graph.

        Args:
            graph: A dictionary, where the keys are the vertices of the graph
            and the values are lists of parents(!) of the key vertices. Setting
            the value to `None` means that the key vertex is a root of the
            graph.

        Examples:

            Initialize a very simple graph with three vertices:

            >>> graph = Graph({'X': None, 'Y': None, 'Z': ['X', 'Y']})

            We can now print a summary of the graph and also draw it:

            >>> graph.summary()
            >>> graph.render()
        """
        # If the input is a dict, assume it is already in the correct format
        if isinstance(graph, (dict, OrderedDict)):
            self.graph = OrderedDict(graph)
            self._restore_order()
        elif isinstance(graph, Graph):
            self.graph = graph.graph
        else:
            raise RuntimeError(f"Could not process {graph} as a graph.")

    def __repr__(self):
        """Define the representation."""
        import pprint
        return pprint.pformat(self.graph)

    def __str__(self):
        """Define the string format."""
        import pprint
        return pprint.pformat(self.graph)

    def __iter__(self):
        """Make the graph iterable."""
        return iter(self.graph)

    def __getitem__(self, item):
        """Expose the internal dictionary representation to the outside."""
        return self.graph[item]

    def _try_add_vertex(self, vertex):
        """Try to add a new vertex to the graph and abort if existent."""
        if vertex in self.graph:
            print("Vertex already exists.")
        else:
            self.graph[vertex] = None
            print("Added vertex ", vertex)

    def _try_add_edge(self, source, target):
        """Try to add a new edge to the graph and abort if existent."""
        if source in self.graph:
            if target not in self.graph[source]:
                self.graph[source].append(target)
            else:
                print("Edge already exists.")
        else:
            self.graph[source] = [target]

    def _restore_order(self):
        """
        Restores a topologically sorted order among the nodes of the graph.
        """
        top_sort = self.topological_sort()
        self.graph = OrderedDict((k, self.graph[k]) for k in top_sort)

    def add_vertices(self, vertices):
        """Add one or multiple vertices to the graph.

        Args:
            vertices: A single hashable object, or an iterable collection
                thereof.
        """
        if isinstance(vertices, (list, tuple)):
            for v in vertices:
                self._try_add_vertex(v)
        else:
            self._try_add_vertex(vertices)
        self._restore_order()

    def add_edge(self, source, target):
        """Add a single edge from source to target."""
        self._try_add_edge(source, target)
        self._restore_order()

    def vertices(self):
        """Find all vertices."""
        return list(self.graph.keys())

    def edges(self):
        """Find all edges."""
        edges = []
        for node, parents in self.graph.items():
            if parents:
                for p in parents:
                    edges.append({p: node})
        return edges

    def roots(self):
        """Find all root vertices."""
        return [node for node in self.graph if not self.graph[node]]

    def non_roots(self):
        """Find all non-root vertices."""
        return [node for node in self.graph if self.graph[node]]

    def leafs(self):
        """Find all leaf vertices."""
        return [v for v in self.vertices() if v not in self.non_leafs()]

    def non_leafs(self):
        """Find all non-leaf vertices."""
        parents = set(sum([p for p in self.graph.values() if p], []))
        return [v for v in self.vertices() if v in parents]

    def parents(self, vertex):
        """Find the parents of a vertex.

        Args:
            vertex: A single vertex of the graph.
        """
        return self.graph[vertex]

    def children(self, vertex):
        """Find the children of a vertex.

        Args:
            vertex: A single vertex of the graph.
        """
        children = []
        for node, parents in self.graph.items():
            if parents and vertex in parents:
                children.append(node)
        return children

    def vertex_number(self):
        """Get the number of vertices."""
        return len(self.graph)

    def edge_number(self):
        """Get the number of edges."""
        return sum([len(v) for k, v in self.graph.items()])

    def descendants(self, vertex):
        """Find all descendants of a vertex.

        Args:
            vertex: A single vertex of the graph.
        """
        descendants = []
        # Start with current children and set exit point for recursion
        current_children = self.children(vertex)
        if not current_children:
            return descendants

        descendants += current_children

        # Recurse down the children
        for child in current_children:
            new_descendants = self.descendants(child)
            descendants += new_descendants

        return list(set(descendants))

    def get_intervened_graph(self, interventions):
        """Return the intervened graph as a new graph.

        Args:
            interventions: Single vertex or an iterable collection of vertices.
        """
        intervened_graph = copy.deepcopy(self.graph)
        if isinstance(interventions, (list, tuple)):
            for i in interventions:
                intervened_graph[i] = None
        else:
            intervened_graph[interventions] = None
        return Graph(intervened_graph)

    def summary(self):
        """Print a detailed summary of the graph."""
        print("Vertices in graph", self.vertices())
        print("Roots in graph", self.roots())
        print("Non-roots in graph", self.non_roots())
        print("Leafs in graph", self.leafs())
        print("Non-leafs in graph", self.non_leafs())
        print("Edges in the graph", self.edges())

        for v in self.vertices():
            print(f"Children of {v} are {self.children(v)}")
            print(f"Parents of {v} are {self.parents(v)}")
            print(f"descendants of {v} are {self.descendants(v)}")

    def _convert_to_nx(self):
        """Convert the graph to a networkx `DiGraph`."""
        nx_graph = nx.DiGraph()
        for edge in self.edges():
            edge = next(iter(edge.items()))
            nx_graph.add_edge(*edge)
        return nx_graph

    def topological_sort(self):
        """Topologically sort the graph through networkx.

        Returns:

            A list of all vertices of the graph sorted in topological order,
            see https://en.wikipedia.org/wiki/Topological_sorting
        """
        nx_graph = self._convert_to_nx()
        return list(nx.topological_sort(nx_graph))

    def render(self, path=None, save=True):
        """Draw the graph with nxpd."""
        from nxpd import draw
        try:
            get_ipython
            from nxpd import nxpdParams
            nxpdParams['show'] = 'ipynb'
        except NameError:
            pass
        nx_graph = self._convert_to_nx()
        nx_graph.graph['dpi'] = 80
        if save:
            filename = os.path.abspath(os.path.join(path, "graph.pdf"))
            draw(nx_graph, filename, format='pdf', show=False)
        else:
            return draw(nx_graph)

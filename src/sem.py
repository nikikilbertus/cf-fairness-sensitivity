"""This module contains a structural equation model class that can be used to
synthesize data from a causal generative model."""

import copy

import numpy as np
from logzero import logger

from graph import Graph


class SEM(Graph):
    """A representation of a structural equation model."""

    def __init__(self, graph):
        """Initialize a structural equation model.

        A structural equation model entails a causal graph. Here, we start from
        the graph and then attach the equations/distributions for each vertex
        in the graph. Hence, `SEM` inherits from `Graph` and is initialized
        with the same argument as a `Graph` object.

        Arguments:

            graph: A dictionary, where the vertices of the graph are the keys
            and each value is a lists of parents(!) of the key vertex. Setting
            the value to `None` means that the vertex is a root of the graph.
        """
        super().__init__(graph)
        self.equations = {}

    def _can_sample(self):
        """Check whether graph and equations are consistent."""
        vertices = set(self.vertices())
        eqs = set(self.equations.keys())
        can_sample = vertices == eqs
        if not can_sample:
            logger.info(f"Vertices: {vertices}, equations for: {eqs}")
        return can_sample

    def attach_equation(self, vertex, equation):
        """Attach an equation or distribution to a vertex.

        In an SEM each vertex is determined by a function of its parents (and
        independent noise), except for root vertices, which follow some
        specified distribution.

        Arguments:

            vertex: The vertex for which we attach the equation.

            equation: A callable with a single argument. For a root vertex,
            this is the number of samples to draw. For non-root vertices the
            argument is a dictionary, where the keys are the parent vertices
            and the values are np.ndarrays containing the data.

        Examples:

            For a root vertex 'X' following a standard normal distribution:

            >>> sem.attach_equation('X', lambda n: np.random.randn(n, 1))

            To attach a standard normal to all root vertices, we can run:

            >>> for v in sem.roots():
            >>>     sem.attach_equation(v, lambda n: np.random.randn(n, 1))

            For a non-root vertex 'Z' that is the sum of its two parents 'X'
            and 'Y', we call:

            >>> sem.attach_equation('Z', lambda data: data['X'] +  data['Y'])
        """
        assert vertex in self.vertices(), f"{vertex} non-existent"
        logger.info(f"Attaching equation to vertex {vertex}...")
        self.equations[vertex] = equation

    def sample(self, n_samples):
        """If possible, sample from the structural equation model.

        We can only sample from the SEM if each vertex has an equation
        attached, the graph is an acyclic DAG and the attached equations are
        consistent with the graph structure.

        Arguments:

            n_samples: The size of the sample to draw.

        Returns:

            The sample as a dictionary with the vertices as keys and np arrays
            as values.
        """
        if not self._can_sample():
            return
        sample = {}
        for v in self.topological_sort():
            logger.info(f"Sample vertex {v}...")
            if v in self.roots():
                sample[v] = self.equations[v](n_samples)
            else:
                sample[v] = self.equations[v](sample)

        for k, v in sample.items():
            sample[k] = np.squeeze(v)
        return sample

    def attach_additive_gauss_poly(self, max_degree=3):
        """Create additive noise model with Gaussian noise an non-linear
        equations."""
        for v in self.topological_sort():
            if v == 'A':
                self.attach_equation(v,
                                     lambda n: 2 * np.random.randint(0, 2,
                                                                     (n, 1)) - 1)
                logger.info("Bernoulli...")
            elif v in self.roots():
                self.attach_equation(v, lambda n: np.random.randn(n, 1))
                logger.info("Standard normal...")
            else:
                parents = sorted(self.parents(v))
                powers = np.random.randint(1, max_degree, (len(parents), 1))
                coeffs = np.random.randn(len(parents), 1)
                iterset = list(zip(coeffs, powers, parents))

                def equation(data, iterset=copy.deepcopy(iterset)):
                    return sum([c * data[parent]**p
                                for c, p, parent in iterset]) + \
                           np.random.randn(len(data[iterset[0][2]]), 1)
                self.attach_equation(v, equation)
                logger.info("Random polynomial...")

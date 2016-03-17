import numpy 
cimport numpy
from libcpp.vector cimport vector

cdef extern from "TFCE_.hxx":
  void tfce[T](float H, float E, float minT, float deltaT, vector[vector[int]]& adjacencyList, T* image, T* enhn)

cdef class Surf:
  cdef vector[vector[int]] *Adjacency

  cdef float H
  cdef float E

  def __init__(self, H, E, pyAdjacency):
#    print("--> initSurf")

    self.H = H
    self.E = E

    cdef vector[vector[int]] *Adjacency_ = new vector[vector[int]]()

    cdef vector[int] Adjacency__
    for i in range(len(pyAdjacency)):
      Adjacency__ = pyAdjacency[i]
      Adjacency_.push_back(Adjacency__)

    self.Adjacency = Adjacency_

  def run(self, numpy.ndarray[float, ndim=1, mode="c"] image, numpy.ndarray[float, ndim=1, mode="c"] enhn):
#    print("--> runSurf")

    tfce[float](self.H, self.E, 0, 0, self.Adjacency[0], &image[0], &enhn[0])

#print("==> initTFCE")


# def callCPP(numpy.ndarray[float, ndim=2, mode="c"] matrix, numpy.ndarray[float, ndim=1, mode="c"] thresholds, numpy.ndarray[float, ndim=2, mode="c"] degree):
#     Graph[float](matrix.shape[0], matrix.shape[1], thresholds.shape[0], &matrix[0, 0], &thresholds[0], &degree[0, 0])

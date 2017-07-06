import numpy 
cimport numpy
from libcpp.vector cimport vector

#    Fast TFCE algorithm using prior adjacency sets
#    Copyright (C) 2016  Lea Waller

#    This program is free software: you can redistribute it and/or modify
#    it under the terms of the GNU General Public License as published by
#    the Free Software Foundation, either version 3 of the License, or
#    (at your option) any later version.

#    This program is distributed in the hope that it will be useful,
#    but WITHOUT ANY WARRANTY; without even the implied warranty of
#    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#    GNU General Public License for more details.

#    You should have received a copy of the GNU General Public License
#    along with this program.  If not, see <http://www.gnu.org/licenses/>.

cdef extern from "fast_tfce.hpp":
  void tfce[T](float H, float E, float minT, float deltaT, vector[vector[int]]& adjacencyList, T* image, T* enhn)

cdef class CreateAdjSet:
  cdef vector[vector[int]] *Adjacency

  cdef float H
  cdef float E

  def __init__(self, H, E, pyAdjacency):

    self.H = H
    self.E = E

    cdef vector[vector[int]] *Adjacency_ = new vector[vector[int]]()

    cdef vector[int] Adjacency__
    for i in range(len(pyAdjacency)):
      Adjacency__ = pyAdjacency[i]
      Adjacency_.push_back(Adjacency__)

    self.Adjacency = Adjacency_

  def run(self, numpy.ndarray[float, ndim=1, mode="c"] image, numpy.ndarray[float, ndim=1, mode="c"] enhn):
    tfce[float](self.H, self.E, 0, 0, self.Adjacency[0], &image[0], &enhn[0])


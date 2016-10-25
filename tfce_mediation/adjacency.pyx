import numpy
cimport numpy
cimport cython

from libcpp.vector cimport vector

#    Create adjacency set based on geodesic distance
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

cdef extern from "geodesic/geodesic_mesh_elements.h" namespace "geodesic":
	cdef cppclass Vertex:
		Vertex() 
	cdef cppclass SurfacePoint:
		SurfacePoint()
		SurfacePoint(Vertex*)

cdef extern from "geodesic/geodesic_mesh.h" namespace "geodesic":
		cdef cppclass Mesh:
				Mesh()
				void initialize_mesh_data[P, F](unsigned num_vertices, P*, unsigned num_faces, F*)
				vector[Vertex]& vertices()

cdef extern from "geodesic/geodesic_algorithm_exact.h" namespace "geodesic":
	cdef cppclass GeodesicAlgorithmExact:
		GeodesicAlgorithmExact(Mesh*)

		void propagate(vector[SurfacePoint]&, double)
		unsigned best_source(SurfacePoint&, double&)


def compute(numpy.ndarray[float, ndim=2, mode="c"] v,
						numpy.ndarray[int, ndim=2, mode="c"] f,
						numpy.ndarray[float, ndim=1, mode="c"] thresholds):
	
	cdef Mesh Mesh_
	Mesh_.initialize_mesh_data[float, int](v.shape[0], &v[0, 0], f.shape[0], &f[0, 0])

	cdef GeodesicAlgorithmExact *Algorithm = new GeodesicAlgorithmExact(&Mesh_)

	cdef vector[SurfacePoint] *Source = new vector[SurfacePoint](1)

	cdef SurfacePoint Target

	cdef double Distance = 0
	cdef double maxDistance = numpy.max(thresholds) + 1e-10 # epsilon

	adjacency = [[[] for i in xrange(v.shape[0])] for k in xrange(thresholds.shape[0])]

	for i in range(0, v.shape[0]):
		Source[0][0] = SurfacePoint(&Mesh_.vertices()[i])
		Algorithm.propagate(Source[0], maxDistance)

		for j in range(i + 1, v.shape[0]):
			Target = SurfacePoint(&Mesh_.vertices()[j])
			Algorithm.best_source(Target, Distance)

			for k in range(thresholds.shape[0]):
				if Distance < thresholds[k]:
					print("--> (" + str(i) + ", " + str(j) + ") " + str(thresholds[k]))
					adjacency[k][j].append(i)
					adjacency[k][i].append(j)

	del Source
	del Algorithm

	return adjacency

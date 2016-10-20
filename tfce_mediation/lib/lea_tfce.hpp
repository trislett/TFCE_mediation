#include <cstdlib>
#include <iostream>
#include <list>
#include <set>
#include <string>
#include <vector>
#include <algorithm>

using namespace std;

template <class RealType>
void tfce(float H, float E, float minT, float deltaT, 
          const vector< vector<int> > & adjacencyList,
          const RealType * __restrict__ image,
          RealType * __restrict__ enhn) {

  int numberOfVertices = adjacencyList.size();

  vector< list<int>* > disjointFind(numberOfVertices);
  set< list<int>* > disjointSets;

  int * __restrict__ imageI = new int[numberOfVertices]; 
  for (int i = 0; i != numberOfVertices; ++i) {
    imageI[i] = i;
  }

  sort(imageI, imageI + numberOfVertices,
        [&image](int i, int j) {
          return image[i] > image[j];
        });

  RealType maxT = image[imageI[0]];

  if (deltaT == 0) {
    deltaT = maxT / 100;
  }

  int j = 0;
  for (float T = maxT; T >= minT; T -= deltaT) { // descending -> incremental connectivity

    while (j < numberOfVertices && image[ imageI[j] ] > T) { // disjoint set algorithm
      disjointFind[ imageI[j] ] = new list<int>(); // make set
      disjointFind[ imageI[j] ]-> push_front(imageI[j]); 

      disjointSets.insert(disjointFind[ imageI[j] ]);

      for (int i = 0; i < adjacencyList[imageI[j]].size(); ++i) {
        int a = adjacencyList[ imageI[j] ][i];
        list<int>* c = disjointFind[ imageI[j] ];

        if (disjointFind[a] && disjointFind[a] != c) {
          for (list<int>::const_iterator iterator = c->begin(); 
               iterator != c->end(); 
               ++iterator) {

            disjointFind[ *iterator ] = disjointFind[a]; // optimize this
          }

          disjointFind[a]->splice(disjointFind[a]->begin(), *c);
  
          disjointSets.erase(c);
          delete c; // free
        }

      }

      ++j;
    }

    float HH = pow(T, H);

    for (set< list<int>* >::const_iterator iterator = disjointSets.begin(); 
         iterator != disjointSets.end(); 
         ++iterator) {
      list<int>* c = *iterator;

      float tfceIncrement = pow(c->size(), E) * HH; 

      for (list<int>::const_iterator iterator_ = c->begin(); 
           iterator_ != c->end(); 
           ++iterator_) {
        enhn[ *iterator_ ] += tfceIncrement; 
      }
    }
  }

  delete [] imageI;

  for (set< list<int>* >::const_iterator iterator = disjointSets.begin(); 
         iterator != disjointSets.end(); 
         ++iterator) {
    delete *iterator;
  }
  // delete [] disjointSets;
}

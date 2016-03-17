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
//  cout << "--> numberOfVertices " << numberOfVertices << endl;

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
//  cout << "--> maxT " << maxT << endl;

  if (deltaT == 0) {
    deltaT = maxT / 100;
  }

  int j = 0;
  for (float T = maxT; T >= minT; T -= deltaT) { // descending -> incremental connectivity
//    cout << "T " << T << endl;

    int j_ = j;
    while (image[ imageI[j_] ] > T && j_ < numberOfVertices) { // disjoint set algorithm
      disjointFind[ imageI[j_] ] = new list<int>(); // make set
      disjointFind[ imageI[j_] ]-> push_front(imageI[j_]); 

      disjointSets.insert(disjointFind[ imageI[j_] ]);

      for (int i = 0; i < adjacencyList[imageI[j_]].size(); ++i) {
        int a = adjacencyList[ imageI[j_] ][i];
        list<int>* c = disjointFind[ imageI[j_] ];

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

      ++j_;
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

    j = j_;
  }

  delete [] imageI;

  for (set< list<int>* >::const_iterator iterator = disjointSets.begin(); 
         iterator != disjointSets.end(); 
         ++iterator) {
    delete *iterator;
  }
  // delete [] disjointSets;
}

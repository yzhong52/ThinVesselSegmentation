#pragma once
#include <iostream>

using namespace std;

class DisjointSet {
private:
	int size;
	int *data;
public:
	DisjointSet( int n_size ) : size( n_size ) {
		data = new int[size];
		memset( data, -1, size*sizeof(int) );
	}
	~DisjointSet(){
		delete[] data;
	}
	inline int find(int id){
		if( data[id] == -1 ) {
			return id;
		} else {
			data[id] = find( data[id] );
			return data[id];
		}
	}
	inline void merge( int id1, int id2 ) { 
		if( data[id1]!=-1 || data[id2]!=-1 ) {
			std::cerr << "Merge Failed: id1 or id2 is not a root node. " << endl;
			return;
		}
		data[id1] = id2;
	}

	friend ostream& operator<<( ostream& out, const DisjointSet& djs );
}; 



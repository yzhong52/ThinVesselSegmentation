#pragma once
#include <iostream>

// Yuchen: Critical Component of Minimum Spanning Tree
// A disjoint-set data structure is a data structure that keeps track of a set of elements
// partitioned into a number of disjoint (nonoverlapping) subsets. A union-find algorithm is
// an algorithm that performs two useful operations on such a data structure.
#include <cstring>

class DisjointSet {
private:
	int size;
	int *data;
public:
	// Constuctors & Desctructors
	DisjointSet( int n_size ) : size( n_size ) {
		data = new int[size];
		memset( data, -1, size*sizeof(int) );
	}
	~DisjointSet(){
		delete[] data;
	}

	int operator[]( const int& i ) const { return data[i]; }

	// Find: Determine which subset a particular element is in. This can be used for determining
	// if two elements are in the same subset.
	inline int find(int id){
		if( data[id] == -1 ) {
			return id;
		} else {
			data[id] = find( data[id] );
			return data[id];
		}
	}
	// Union: Join two subsets into a single subset
	inline void merge( int id1, int id2 ) {
		if( data[id1]!=-1 ) id1 = find(id1);
		if( data[id2]!=-1 ) id2 = find(id2);
		if( id1==id2 ) return;
		data[id1] = id2;
	}

	friend std::ostream& operator<<( std::ostream& out, const DisjointSet& djs );
};



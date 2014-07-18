#include "DisjointSet.h"

#include <stdio.h>
#include <string.h>

DisjointSet::DisjointSet( int n_size ) : size( n_size )
{
    data = new int[size];
    memset( data, -1, size*sizeof(int) );
}

DisjointSet::~DisjointSet()
{
    delete[] data;
}

std::ostream& operator<<( std::ostream& out, const DisjointSet& djs )
{
    for( int i=0; i<djs.size; i++ ) out <<  djs.data[i] << " ";
    return out;
}

#include "DisjointSet.h"


ostream& operator<<( ostream& out, const DisjointSet& djs ){
	for( int i=0; i<djs.size; i++ ) out <<  djs.data[i] << " ";
	return out;
}
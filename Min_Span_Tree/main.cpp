// Min_Span_Tree.cpp : Defines the entry point for the console application.
//

#include "MinSpanTree.h"
#include "GLViewer.h"
#include "DataTypes.h"
#include <iostream>
using namespace std;

int main(int argc, char* argv[])
{
	Graph<Edge_Ext, LineSegment> tree;
	MinSpanTree::build_tree_xuefeng( "data/roi16.partial.linedata.txt", tree );
	
	/////////////////////////////////////////
	// Visualize the Data
	/////////////////////////////////////////
	int sx = 134;
	int sy = 113;
	int sz = 116;
	int size = sx * sy * sz;

	// loading data from file
	unsigned char* data = new unsigned char[ size ];
	FILE* pFile=fopen( "data/roi15.uchar.data", "rb" );
	if( pFile==0 ) {
		cout << "File not found" << endl;
		return false;
	}
	int size_read = (int) fread( data, sizeof(unsigned char), size, pFile);


	GLViewer::MIP( data, sx, sy, sz, tree );

	delete[] data;

	return 0;
}


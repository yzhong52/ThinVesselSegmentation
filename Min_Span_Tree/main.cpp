// Min_Span_Tree.cpp : Defines the entry point for the console application.
//

#include "stdafx.h"
#include "Graph.h"
#include "Edge_Graph.h"
#include "Edge.h"
#include "DisjointSet.h"
#include <queue>
#include <iostream>
#include <fstream>
#include "GLViewer.h"
using namespace std;


int _tmain(int argc, _TCHAR* argv[])
{
	//// Test Case
	//// [1] --3-- [2]
	////  |       / | \
	////  |     /   |   \
	////  7    2    4    6
	////  |  /      |      \
	////  |/        |        \
	//// [3] --1-- [4] --5-- [0]
	//// Result
	//// [1] --3-- [2]
	////          /    
	////        /        
	////       2         
	////     /              
	////   /                  
	//// [3] --1-- [4] --5-- [0]
	////// Build Graph
	//Edge_Graph<Edge> original( 5 );
	//original.add_edge( Edge(1, 2, 3) );
	//original.add_edge( Edge(1, 3, 7) );
	//original.add_edge( Edge(2, 3, 2) );
	//original.add_edge( Edge(2, 4, 4) );
	//original.add_edge( Edge(2, 0, 6) );
	//original.add_edge( Edge(3, 4, 1) );
	//original.add_edge( Edge(4, 0, 5) );
	////// Compute Minimum Spanning Tree
	//Edge_Graph<Edge> result;
	//get_min_span_tree( original, result );
	////// Print Result
	//cout << original << endl;
	//cout << result << endl;

	std::ifstream fin("data/linedata.txt");
	if( !fin.is_open() ){
		std::cerr << "File cannot be open" << endl;
	}

	int num_line;
	
	
	fin >> num_line;
	vector<Line> lines(num_line); 
	for( int i=0; i<num_line; i++ ) {
		fin >> lines[i].p1.x;
		fin >> lines[i].p1.y;
		fin >> lines[i].p1.z;
		fin >> lines[i].p2.x;
		fin >> lines[i].p2.y;
		fin >> lines[i].p2.z;

		int num_points;
		fin >> num_points;
		for( int j=0; j<num_points+1; j++ ) {
			fin.ignore(256,'\n');
		}
	}

	

	// build the graph
	Edge_Graph<Edge_Ext> g( num_line );
	for( int i=0; i<num_line; i++ ) {
		for( int j=i+1; j<num_line; j++ ){
			///////////////////////////////////////
			// calculate distance between line
			/////////////////
			// The four endpoints of the line
			Vec3f p1 = lines[i].p1;
			Vec3f p11 = lines[i].p2;
			Vec3f p2 = lines[j].p1;
			Vec3f p22 = lines[j].p2;
			// line
			Vec3f l1 = p11 - p1;
			Vec3f l2 = p22 - p2;
			// A line that is intesect with both l1 and l2
			//     Vec3f l = [ p1*t1 + p11*(1-t1) ] - [ p2*t2 + p2*(1-t2) ]
			//         =-t(p11-p1) + s(p22-p2) - (p22-p11)
			//         = -t * l1 + s * l2 - (p22-p11)
			// l is pependicular to both l1 and l2
			//     l.t() DOT_PRODUCT l1 = 0
			//     l.t() DOT_PRODUCT l2 = 0
			// we have 
			//     -||p11-p1||^2     t1  +  <p22-p2, p11-p1>  t2  -  <p22-p11, p11-p1> = 0
			//     -<p11-p1, p22-p2> t1  +    ||p22-p2||^2    t2  -  <p22-p11, p22-p2> = 0
			// Or
			//     -<l1, l1> * t1 + <l2, L1> t2 - <p22-p11, l1> = 0
			//     -<l1, L2> * t1 + <l2, L2> t2 - <p22-p11, l2> = 0
			
			float a11 = -l1.dot( l1 );
			float a12 =  l2.dot( l1 );
			float a21 = -l1.dot( l2 );
			float a22 =  l2.dot( l2 );
			float b1 = (p22-p11).dot( l1 );
			float b2 = (p22-p11).dot( l2 );

			float denominator = a22 * a11 - a12 * a21;

			Edge_Ext e;
			e.node1 = i;
			e.node2 = j;

			float temp = (p1 -p2 ).dot(p1 -p2 );
			e.weight = temp;
			e.p1 = p1;
			e.p2 = p2;

			temp = (p11-p2 ).dot(p11-p2 );
			if( temp < e.weight ) { 
				e.weight = temp; 
				e.p1 = p11; 
				e.p2 = p2; 
			}

			temp = (p1 -p22).dot(p1 -p22);
			if( temp < e.weight ) 
			{
				e.weight = temp;
				e.p1 = p1;
				e.p2 = p22;
			}


			temp = (p11-p22).dot(p11-p22);
			if( temp < e.weight ) {
				e.weight = temp;
				e.p1 = p11;
				e.p2 = p22;
			}
			// if( temp > 190 ) continue;

			g.add_edge( e );
		}
	}

	Edge_Graph<Edge_Ext> tree;
	g.get_min_span_tree( tree );
	// cout << tree << endl;


	// Visualization of MIP
	int sx = 111;
	int sy = 44;
	int sz = 111;
	int size = sx * sy * sz;

	unsigned char* data = new unsigned char[ size ];

	// loading data from file
	FILE* pFile=fopen( "data/roi16.partial.partial.uchar.data", "rb" );
	if( pFile==0 ) {
		cout << "File not found" << endl;
		return false;
	}

	int size_read = (int) fread( data, sizeof(unsigned char), size, pFile);
	
	GLViewer::MIP( data, sx, sy, sz, tree, lines );

	delete[] data;


	return 0;
}


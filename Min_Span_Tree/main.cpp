// Min_Span_Tree.cpp : Defines the entry point for the console application.
//

#include "stdafx.h"
#include "Graph.h"
#include "Edge_Graph.h"
#include "Edge.h"
#include "DisjointSet.h"
#include <queue>
#include <iostream>
using namespace std;

template<class Edge_Type>
void get_min_span_tree(const Edge_Graph<Edge_Type>& src, Edge_Graph<Edge_Type>& dst )
{
	dst.reset( src.get_num_nodes() );
	DisjointSet djs( src.get_num_nodes() );

	std::priority_queue<Edge_Type> edges = src.get_edges();
	while( !edges.empty() /*&& dst.get_num_edges()<dst.get_num_nodes()-1*/ ) {
		Edge e = edges.top();
		int sid1 = djs.find( e.node1 );
		int sid2 = djs.find( e.node2 );
		if( sid1 != sid2 ) {
			dst.add_edge( e ); 
			djs.merge( sid1, sid2 );
		}
		edges.pop(); 
	}
}

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

	return 0;
}


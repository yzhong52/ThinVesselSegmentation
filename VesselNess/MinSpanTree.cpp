#include "MinSpanTree.h"

#include <iostream>
#include <fstream>
#include "Graph.h"
#include "Edge.h"
#include "DisjointSet.h"
#include "DataTypes.h"

using namespace std;

namespace MinSpanTree
{
	inline int build_tree_example( ){
		// Yuchen: Minimum Spanning Tree

		// Test Case
		// [1] --3-- [2]
		//  |       / | \
		//  |     /   |   \
		//  7    2    4    6
		//  |  /      |      \
		//  |/        |        \
		// [3] --1-- [4] --5-- [0]
		// Result
		// [1] --3-- [2]
		//          /    
		//        /        
		//       2         
		//     /              
		//   /                  
		// [3] --1-- [4] --5-- [0]
		//// Build Graph
		Graph<Edge> graph( 5 );
		graph.add_edge( Edge(1, 2, 3) );
		graph.add_edge( Edge(1, 3, 7) );
		graph.add_edge( Edge(2, 3, 2) );
		graph.add_edge( Edge(2, 4, 4) );
		graph.add_edge( Edge(2, 0, 6) );
		graph.add_edge( Edge(3, 4, 1) );
		graph.add_edge( Edge(4, 0, 5) );

		// Compute Minimum Spanning Tree
		Graph<Edge> tree( graph.num_nodes() );
		DisjointSet djs( graph.num_nodes() );

		std::priority_queue<Edge> edges = graph.get_edges();
		while( !edges.empty() && tree.num_edges()<tree.num_nodes()-1 ) {
			Edge e = edges.top();
			int sid1 = djs.find( e.node1 );
			int sid2 = djs.find( e.node2 );
			if( sid1 != sid2 ) {
				tree.add_edge( e ); 
				djs.merge( sid1, sid2 );
			}
			edges.pop(); 
		}

		//// Print Result
		cout << graph << endl;
		cout << tree << endl;
	}

	bool build_tree_xuefeng( const std::string& file_name, 
		/*Output*/ Graph<Edge_Ext, LineSegment>& tree,
		float thres ) {
		Graph<Edge_Ext, LineSegment> graph;

		/////////////////////////////////////////////////////////////
		// Loading Data
		/////////////////////////////////////////////////////////////
		// Yuchen: I am working with my collegue Xuefeng. 
		// The data are supposed to be a bunch of line segments in 3D
		int num_line;
		// Open File
		std::ifstream fin;
		fin.open( file_name.c_str() );
		if( !fin.is_open() ){
			std::cerr << "File cannot be open" << endl;
		}
		// Reading Data from file
		fin >> num_line;
		graph.reset( num_line );
		for( int i=0; i<num_line; i++ ) {
			// each line segments are defined as two end points in 3d
			fin >> graph.get_node(i).p1.x; 
			fin >> graph.get_node(i).p1.y;
			fin >> graph.get_node(i).p1.z;
			fin >> graph.get_node(i).p2.x;
			fin >> graph.get_node(i).p2.y;
			fin >> graph.get_node(i).p2.z;
			// There are some surrounding points around.
			// I am just going to ignore them for now.
			int num_points;
			fin >> num_points;
			for( int j=0; j<num_points+1; j++ ) {
				fin.ignore(256,'\n');
			}
		}

		///////////////////////////////////////////////////////////////
		// build the graph
		///////////////////////////////////////////////////////////////
		
		for( int i=0; i<num_line; i++ ) {
			for( int j=i+1; j<num_line; j++ ){
				///////////////////////////////////////
				// calculate distance between line
				/////////////////
				// The four endpoints of the line
				Edge_Ext e;
				e.node1 = i;
				e.node2 = j;
				
				float weight = 0;
				Vec3f to;

				// Case 1
				graph.get_node(i).get_distance( graph.get_node(j).p1, to, weight );
				e.weight = weight;
				e.line.p1 = graph.get_node(j).p1;
				e.line.p2 = to;
				// Case 2
				graph.get_node(i).get_distance( graph.get_node(j).p2, to, weight );
				if( e.weight > weight ){
					e.weight = weight;
					e.line.p1 = graph.get_node(j).p2;
					e.line.p2 = to; 
				}
				// Case 3
				graph.get_node(j).get_distance( graph.get_node(i).p1, to, weight );
				if( e.weight > weight ){
					e.weight = weight;
					e.line.p1 = graph.get_node(i).p1;
					e.line.p2 = to; 
				}
				// Case 4
				graph.get_node(j).get_distance( graph.get_node(i).p2, to, weight );
				if( e.weight > weight ){
					e.weight = weight;
					e.line.p1 = graph.get_node(i).p2;
					e.line.p2 = to; 
				}

				if( e.weight < thres ) 
					graph.add_edge( e );
			}
		}

		graph.get_min_span_tree( tree );
		// cout << tree << endl;
		return true;
	}
}

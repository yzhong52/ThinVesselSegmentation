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


	bool load_graph( Graph<Edge_Ext, LineSegment>& graph, const std::string& file_name ) {
		/////////////////////////////////////////////////////////////
		// Loading Data
		/////////////////////////////////////////////////////////////
		// Yuchen: I am working with my collegue Xuefeng. 
		// The data are supposed to be a bunch of line segments in 3D
		int num_line1, num_line2;
		// Open File
		std::ifstream fin1, fin2;
		string filename1 = file_name;
		string filename2 = file_name;
		filename1 += ".linedata.txt";
		filename2 += ".models.txt";

		fin1.open( filename1 );
		fin2.open( filename2 );
		if( !fin1.is_open() ){
			std::cerr << "File cannot be open" << endl;
			return 0;
		}
		if( !fin2.is_open() ){
			std::cerr << "File cannot be open" << endl;
			return 0;
		}

		
		// Reading Data from file
		fin1 >> num_line1;
		fin2 >> num_line2;
		if( num_line1 != num_line2 ){
			cout << "Data Does not match" << endl;
			return 0;
		}
		int& num_line = num_line1;
		graph.reset( num_line1 );
		for( int i=0; i<num_line1; i++ ) {
			// each line segments are defined as two end points in 3d
			fin1 >> graph.get_node(i).p1.x; 
			fin1 >> graph.get_node(i).p1.y; 
			fin1 >> graph.get_node(i).p1.z; 
			fin1 >> graph.get_node(i).p2.x; 
			fin1 >> graph.get_node(i).p2.y; 
			fin1 >> graph.get_node(i).p2.z; 
			
			float temp;
			fin2 >> temp; if( temp!= graph.get_node(i).p1.x ) { cout << "Error: Data does not match" << endl; return 0; };
			fin2 >> temp;
			fin2 >> temp;
			fin2 >> temp;
			fin2 >> temp;
			fin2 >> temp;
			fin2 >> graph.get_node(i).radius;
			// There are some surrounding points around.
			// I am just going to ignore them for now.
			int num_points;
			fin1 >> num_points;
			for( int j=0; j<num_points; j++ ) {
				MST::Point3D p;
				fin1 >> p.x;
				fin1 >> p.y;
				fin1 >> p.z;
				graph.get_node(i).points.push_back( p );
			}
		}
		fin1.close();
		fin2.close();
		return true; 
	}

	bool save_graph( Graph<Edge_Ext, LineSegment>& graph, const std::string& file_name ) {
		/////////////////////////////////////////////////////////////
		// Saving Data
		/////////////////////////////////////////////////////////////
		// Yuchen: I am working with my collegue Xuefeng. 
		// The data are supposed to be a bunch of line segments in 3D
		
		std::ofstream fout1, fout2; // Open File
		string filename1 = file_name;
		string filename2 = file_name;
		filename1 += ".linedata.txt";
		filename2 += ".models.txt";

		fout1.open( filename1 );
		fout2.open( filename2 );
		if( !fout1.is_open() ){
			std::cerr << "File cannot be open" << endl;
			return 0;
		}
		if( !fout2.is_open() ){
			std::cerr << "File cannot be open" << endl;
			return 0;
		}

		
		// Reading Data from file
		fout1 << graph.num_nodes() << endl; 
		fout2 << graph.num_nodes() << endl; 

		for( unsigned int i=0; i<graph.num_nodes(); i++ ) {
			// each line segments are defined as two end points in 3d
			fout1 << graph.get_node(i).p1.x << " "; 
			fout1 << graph.get_node(i).p1.y << " "; 
			fout1 << graph.get_node(i).p1.z << " "; 
			fout1 << graph.get_node(i).p2.x << " "; 
			fout1 << graph.get_node(i).p2.y << " "; 
			fout1 << graph.get_node(i).p2.z << endl; 
			fout1 << graph.get_node(i).points.size() << endl;
			for( unsigned int j=0; j<graph.get_node(i).points.size(); j++ ) {
				MST::Point3D& p = graph.get_node(i).points[j];
				fout1 << p.x << " "; 
				fout1 << p.y << " "; 
				fout1 << p.z << endl; 
			}

			fout2 << graph.get_node(i).p1.x << " "; 
			fout2 << graph.get_node(i).p1.y << " "; 
			fout2 << graph.get_node(i).p1.z << " "; 
			fout2 << graph.get_node(i).p2.x << " "; 
			fout2 << graph.get_node(i).p2.y << " "; 
			fout2 << graph.get_node(i).p2.z << " "; 
			fout2 << graph.get_node(i).radius << endl;
		}
		fout1.close();
		fout2.close();
		return true;
	}

	bool pre_process_xuefeng( const std::string& file_name,
		const std::string& save_file_name, 
		/*Output*/ Graph<Edge_Ext, LineSegment>& ring, 
		/*INPUT*/  Vec3f center_of_ring )
	{
		Graph<Edge_Ext, LineSegment> graph;
		Graph<Edge_Ext, LineSegment> graph_no_rings; 

		// loading data from file
		load_graph( graph, file_name );
		
		///////////////////////////////////////////////////////////////
		// processing - rings reduction 
		///////////////////////////////////////////////////////////////
		// Rings criteria
		// 1) parallel to x-y plane
		// 2) sigma is maller than (or equals to) 0.5
		// 3) perpendicular to the radical direction of the rings (we know the center of the ring) 
		
		for( unsigned int i=0; i<graph.num_nodes(); i++ ) {
			MST::LineSegment& line = graph.get_node(i);
			const Vec3f& p1 = line.p1;
			const Vec3f& p2 = line.p2;

			bool isRing = false;
			if( abs(p1.z-p2.z)/((p1-p2).length()) < 0.1 ) {
				if( line.radius <= 75.0f /*Not so usefully at the moment*/) {
					// the center point of the line segment
					const Vec3f p3 = (p1 + p2) * 0.5f;
					Vec3f dir_line = p1 - p2;
					Vec3f dir_radical = p3 - center_of_ring;
					dir_line.z = dir_radical.z = 0;

					float dotproduct = dir_line.dot( dir_radical ); 
					dotproduct /= dir_line.length();
					dotproduct /= dir_radical.length();
					if( abs(dotproduct) < 0.02 ) {
						isRing = true; 
					}
				}
			}

			if( isRing ) {
				ring.add_node( line );
			} else {
				graph_no_rings.add_node( line );
			}
		}

		// save the graph without rings
		save_graph( graph_no_rings, save_file_name );

		return true;
	}

	bool build_tree_xuefeng( const std::string& dataname, 
		/*Output*/ Graph<Edge_Ext, LineSegment>& tree,
		/*Input*/ float thres ) 
	{
		Graph<Edge_Ext, LineSegment> graph;
		// load_graph( graph, dataname );

		/////////////////////////////////////////////////////////////
		// Loading Data
		/////////////////////////////////////////////////////////////
		// Yuchen: I am working with my collegue Xuefeng. 
		// The data are supposed to be a bunch of line segments in 3D
		int num_line1, num_line2;
		// Open File
		std::ifstream fin1, fin2;
		string filename1 = dataname;
		string filename2 = dataname;
		filename1 += ".linedata.txt";
		filename2 += ".models.txt";

		fin1.open( filename1 );
		fin2.open( filename2 );
		if( !fin1.is_open() ){
			std::cerr << "File cannot be open" << endl;
			return 0;
		}
		if( !fin2.is_open() ){
			std::cerr << "File cannot be open" << endl;
			return 0;
		}

		// Reading Data from file
		fin1 >> num_line1;
		fin2 >> num_line2;
		if( num_line1 != num_line2 ){
			cout << "Data Does not match" << endl;
			return 0;
		}
		int& num_line = num_line1;
		graph.reset( num_line1 );
		for( int i=0; i<num_line1; i++ ) {
			// each line segments are defined as two end points in 3d
			fin1 >> graph.get_node(i).p1.x; 
			fin1 >> graph.get_node(i).p1.y; 
			fin1 >> graph.get_node(i).p1.z; 
			fin1 >> graph.get_node(i).p2.x; 
			fin1 >> graph.get_node(i).p2.y; 
			fin1 >> graph.get_node(i).p2.z; 
			
			float temp;
			fin2 >> temp; if( temp!= graph.get_node(i).p1.x ) { cout << "Error: Data does not match" << endl; return 0; };
			fin2 >> temp;
			fin2 >> temp;
			fin2 >> temp;
			fin2 >> temp;
			fin2 >> temp;
			fin2 >> graph.get_node(i).radius;
			// There are some surrounding points around.
			// I am just going to ignore them for now.
			int num_points;
			fin1 >> num_points;
			for( int j=0; j<num_points; j++ ) {
				MST::Point3D p;
				fin1 >> p.x;
				fin1 >> p.y;
				fin1 >> p.z;
				graph.get_node(i).points.push_back( p );
			}
		}

		///////////////////////////////////////////////////////////////
		// build the graph (adding edges)
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
				weight -= graph.get_node(j).radius + graph.get_node(i).radius;
				e.line.p1 = graph.get_node(j).p1;
				e.line.p2 = to;

				// Case 2
				graph.get_node(i).get_distance( graph.get_node(j).p2, to, weight );
				weight -= graph.get_node(j).radius + graph.get_node(i).radius;
				if( e.weight > weight ){
					e.weight = weight;
					e.line.p1 = graph.get_node(j).p2;
					e.line.p2 = to; 
				}
				// Case 3
				graph.get_node(j).get_distance( graph.get_node(i).p1, to, weight );
				weight -= graph.get_node(j).radius + graph.get_node(i).radius;
				if( e.weight > weight ){
					e.weight = weight;
					e.line.p1 = graph.get_node(i).p1;
					e.line.p2 = to; 
				}
				// Case 4
				graph.get_node(j).get_distance( graph.get_node(i).p2, to, weight );
				weight -= graph.get_node(j).radius + graph.get_node(i).radius;
				if( e.weight > weight ){
					e.weight = weight;
					e.line.p1 = graph.get_node(i).p2;
					e.line.p2 = to; 
				}

				if( e.weight < thres ) {
					graph.add_edge( e );
				}
			}
		}

		graph.get_min_span_tree( tree );
		// cout << tree << endl;
		return true;
	}
}

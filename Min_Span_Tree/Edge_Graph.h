#pragma once

#include <queue>
#include <iostream>
#include "Graph.h" 
#include "Edge.h"


template<class Edge_Type>
class Edge_Graph : public Graph
{
	// private variables
	std::priority_queue<Edge> edges;

public:
	Edge_Graph( unsigned int n_size );
	Edge_Graph( void ){ };
	~Edge_Graph( void );
	void reset( int n_size );
	
	// add an edge to a graph
	void add_edge( Edge edge ){
		edges.push( edge ); 
	}

	//gettters
	const std::priority_queue<Edge_Type>& get_edges( void ) const { return edges; }
	const unsigned int get_num_edges(void) const { return edges.size(); }
	
	template<class T>
	friend std::ostream& operator<<( std::ostream& out, Edge_Graph<T>& g );
};


template<class Edge_Type>
Edge_Graph<Edge_Type>::Edge_Graph( unsigned int num_node )
	: Graph( num_node )
{

}


template<class Edge_Type>
Edge_Graph<Edge_Type>::~Edge_Graph(void)
{

}

template<class Edge_Type>
void Edge_Graph<Edge_Type>::reset( int num_node ){
	Graph::reset( num_node );
	while( !edges.empty() ) edges.pop();
}

template<class T>
std::ostream& operator<<( std::ostream& out, Edge_Graph<T>& g ) {
	if( g.edges.empty() ) {
		out << "There is currently no edges in graph :( ";
		return out;
	}
	T* pt_edge = &g.edges.top();
	for( unsigned int i=0; i<g.edges.size(); i++ ) {
		cout << (*pt_edge) << endl;
		pt_edge++;
	}
	return out;
}

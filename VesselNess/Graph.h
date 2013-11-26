#pragma once

#include <iostream>
#include <queue>
#include <vector>
#include "Edge.h"
#include "DisjointSet.h" // for Minimum Spaning Tree



template<class Edge_Type = Edge, class Node_Type = char>
class Graph
{
private:
	// number of nodes
	std::vector<Node_Type> nodes;
	// Edges of the graph
	std::priority_queue<Edge_Type> edges;
public:
	// Constructor & Destructor
	Graph( unsigned int num_node ){  nodes.resize( num_node ); }
	Graph( Graph& g ) : nodes( g.nodes), edges( g.edges ) { }
	Graph( void ){ };
	~Graph( void ){ }

	// reset the graph and clear all the edges
	void reset( int num_node ) {
		// resize the nodes
		nodes.resize( num_node );
		// clear the edges
		while( !edges.empty() ) edges.pop();
	}
	
	// add an edge to a graph
	void add_edge( Edge_Type edge ){
		edges.push( edge ); 
	}

	//gettters
	inline const std::priority_queue<Edge_Type>& get_edges( void ) const { return edges; }
	inline       std::priority_queue<Edge_Type>& get_edges( void )       { return edges; }
	inline const std::vector<Node_Type>& get_nodes( void ) const { return nodes; }
	inline       std::vector<Node_Type>& get_nodes( void )       { return nodes; }
	inline Node_Type& get_node(const int& i){ return nodes[i]; }
	const unsigned int num_edges(void) const { return (unsigned int) edges.size(); }
	const unsigned int num_nodes(void) const { return (unsigned int) nodes.size(); }

	// get a minimum spaning tree of the current graph
	void get_min_span_tree( Graph<Edge_Type, Node_Type>& dst ) const;

	// print graph for debuging
	template<class E, class N>
	friend std::ostream& operator<<( std::ostream& out, Graph<E, N>& g );
};


template<class Edge_Type, class Node_Type>
void Graph<Edge_Type, Node_Type>::get_min_span_tree( Graph<Edge_Type, Node_Type>& dst )  const
{
	dst.reset( this->num_nodes() );
	DisjointSet djs( this->num_nodes() );

	// build edges
	std::priority_queue<Edge_Type> edges = this->get_edges();
	while( !edges.empty() && dst.num_edges()<dst.num_nodes()-1 ) {
		Edge_Type e = edges.top();
		int sid1 = djs.find( e.node1 );
		int sid2 = djs.find( e.node2 );
		if( sid1 != sid2 ) {
			dst.add_edge( e ); 
			djs.merge( sid1, sid2 );
		}
		edges.pop(); 
	}

	// copy nodes
	dst.get_nodes() = this->get_nodes();
}

template<class E, class N>
std::ostream& operator<<( std::ostream& out, Graph<E, N>& g ) {
	if( g.edges.empty() ) {
		out << "OMG: There is currently no edges in graph.";
		return out;
	} else {
		out << "Number of Edge: " << g.num_edges() << endl;
	}
	// Tranverse the edges and print them
	E* pt_edge = &g.edges.top();
	for( unsigned int i=0; i<g.edges.size(); i++ ) {
		cout << (*pt_edge) << endl;
		pt_edge++;
	}
	return out;
}


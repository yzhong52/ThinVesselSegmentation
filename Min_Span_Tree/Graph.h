#pragma once
class Graph
{
	unsigned int node_size; // number of nodes
public:
	// constructor
	Graph( unsigned int num_of_nodes = 0) : node_size( num_of_nodes ) {} 

	// getters
	unsigned int get_num_nodes( void ) const { return node_size; }
	
	virtual const unsigned int get_num_edges(void) const = 0;
	// resize
	void reset( int num_of_nodes = 0 ) { node_size = num_of_nodes; } 
};


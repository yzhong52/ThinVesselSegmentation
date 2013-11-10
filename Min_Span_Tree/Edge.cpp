#include "Edge.h"


Edge::Edge(void)
{
}


Edge::~Edge(void)
{
}



void print(const Edge& e){
	std::cout << "Hello"; 
}

std::ostream& operator<<( std::ostream& out, const Edge& e ) { 
	out << " - Node: " << e.node1 << ", " << e.node2 << " Weight: " << e.weight;
	return out;
}
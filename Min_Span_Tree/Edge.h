#pragma once
#include <iostream>

class Edge
{
public:
	Edge(void);
	~Edge(void);

	int node1, node2;
	float weight;

	// operator overflow
	bool operator<( Edge right ) const {
		return (this->weight > right.weight);
	}

	Edge( int node1, int node2, float weight ) 
		: node1( node1 ), node2( node2 ), weight( weight ) { }

	friend std::ostream& operator<<( std::ostream& out, const Edge& e );
	friend void print(const Edge& e);
};

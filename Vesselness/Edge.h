#pragma once
#include <iostream>

class Edge
{
public:
    Edge( int node1 = -1, int node2 = -1, float weight = -1 )
        : node1( node1 ), node2( node2 ), weight( weight ) { }
    ~Edge(void) {};

    int node1, node2;
    float weight;

    // operator overflow
    inline bool operator<( Edge right ) const
    {
        return (this->weight > right.weight);
    }

    friend std::ostream& operator<<( std::ostream& out, const Edge& e );
};


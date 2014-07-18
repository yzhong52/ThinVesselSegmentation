#ifndef MST_EDGE
#define MST_EDGE
#include <iostream>

namespace MST
{

class Edge
{
public:
    // An edge is defined by two node (their id) and the weight between
    int node1, node2;
    float weight;

    Edge( int node1 = -1, int node2 = -1, float weight = -1 );

    // compare operator overflow
    inline bool operator<( Edge right ) const;
    inline friend std::ostream& operator<<( std::ostream& out, const Edge& e );
};

inline bool Edge::operator<( Edge right ) const
{
    return (this->weight > right.weight);
}

std::ostream& operator<<( std::ostream& out, const Edge& e )
{
    out << " - Node: " << e.node1 << ", " << e.node2 << " Weight: " << e.weight;
    return out;
}

} // end of namespace

#endif // MST_EDGE


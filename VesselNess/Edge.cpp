#include "Edge.h"

std::ostream& operator<<( std::ostream& out, const Edge& e )
{
    out << " - Node: " << e.node1 << ", " << e.node2 << " Weight: " << e.weight;
    return out;
}

#include "MSTEdgeExt.h"

namespace MST
{

EdgeExt::EdgeExt( int node1, int node2, float weight, float s )
    : Edge( node1, node2, weight)
    , sigma( s )
{
    //ctor
}

EdgeExt::~EdgeExt()
{
    //dtor
}

} // end of namespace MST

#ifndef MSTEDGEEXT_H
#define MSTEDGEEXT_H

#include "MSTEdge.h"

namespace MST
{

class EdgeExt : public Edge
{
    public:
        EdgeExt( int node1 = -1, int node2 = -1,
                 float weight = -1, float s = -1.0f );
        virtual ~EdgeExt();

    private:
        float sigma;
};

} // end of namespace MST

#endif // MSTEDGEEXT_H

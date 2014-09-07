#ifndef GLMINSPANTREE_H
#define GLMINSPANTREE_H

#include <unordered_map>
#include <opencv2/core/core.hpp>

#include "GLViewer.h"
#include "MSTGraph.h"
#include "MSTEdgeExt.h"

namespace GLViewer
{

class GLMinSpanTree : public GLViewer::Object
{
public:
    GLMinSpanTree( const MST::Graph<MST::EdgeExt, cv::Vec3d>& graph,
                   const DisjointSet& disjointset,
                   const cv::Vec3i& volume_size,
                   const int number_of_highlisht_branches = 2 );

    virtual ~GLMinSpanTree();

    virtual void render( void );

    // size of the object
    virtual unsigned int size_x(void) const
    {
        return size[0];
    }
    virtual unsigned int size_y(void) const
    {
        return size[1];
    }
    virtual unsigned int size_z(void) const
    {
        return size[2];
    }

    virtual void init( void )
    {
        glDisable(GL_DEPTH_TEST);
        glEnable(GL_BLEND);
        glBlendFunc(GL_ONE, GL_ONE);
        glBlendEquation( GL_MAX_EXT );
    }
private:
    const MST::Graph<MST::EdgeExt, cv::Vec3d>& graph;
    const DisjointSet& djs;
    const cv::Vec3i& size;

    std::unordered_map<int, cv::Vec3b> colors;
};

}// end of namespace

#endif // GLMINSPANTREE_H

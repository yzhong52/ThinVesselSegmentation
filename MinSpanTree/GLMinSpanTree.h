#ifndef GLMINSPANTREE_H
#define GLMINSPANTREE_H

#include <opencv2/core/core.hpp>

#include "GLViewer.h"
#include "MSTGraph.h"

namespace GLViewer
{

class GLMinSpanTree : public GLViewer::Object
{
    const MST::Graph<MST::Edge, cv::Vec3d>& graph;
    const cv::Vec3i& size;

    cv::Vec3f color;
public:
    GLMinSpanTree( const MST::Graph<MST::Edge, cv::Vec3d>& graph,
                   const cv::Vec3i& size );
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

    inline void set_color( cv::Vec3f new_color )
    {
        color = new_color;
    }

};


}// end of namespace

#endif // GLMINSPANTREE_H

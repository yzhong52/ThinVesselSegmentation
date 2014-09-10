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
                   const int& min_num_nodes = 2,
                   const int& max_num_highlights = 25 );

    virtual ~GLMinSpanTree();

    virtual void render( void );

    // volumn size of the object
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

    virtual void keyboard( unsigned char k );

private:
    typedef cv::Vec3b Color3b;

    struct MapElement
    {
        int rank;
        Color3b color;
        MapElement( const int& r, const Color3b& c ) : rank(r), color(c) {}
    };

    const MST::Graph<MST::EdgeExt, cv::Vec3d>& graph;
    const DisjointSet& djs;

    // Size of the volumn
    const cv::Vec3i& size;

    std::unordered_map<int, MapElement> color_map;

    float sigma_min, sigma_max, sigma_range;

    /* 'branch_display' indicates which branch to show. For example,
           "branch_display==0": branch i will be displayed. However,
        the three most important branches are alwasy shown. */
    int branch_display;
};

}// end of namespace

#endif // GLMINSPANTREE_H

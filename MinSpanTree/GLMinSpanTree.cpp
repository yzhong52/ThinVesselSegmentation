#include "GLMinSpanTree.h"
#include <opencv2/core/core.hpp>

using namespace cv;
using namespace MST;

namespace GLViewer
{

GLMinSpanTree::GLMinSpanTree( const MST::Graph<MST::Edge, cv::Vec3d>& g,
                              const cv::Vec3i& s )
    : graph( g ), size( s )
{



}



GLMinSpanTree::~GLMinSpanTree()
{

}



void GLMinSpanTree::render( void )
{
    glColor3f( 1.0f, 0.4f, 0.4f );
    glBegin( GL_LINES );
    for( unsigned i=0; i<graph.num_edges(); i++ )
    {
        const Edge& e = graph.get_edge( i );
        const Vec3d& p1 = graph.get_node( e.node1 );
        const Vec3d& p2 = graph.get_node( e.node2 );
        glVertex3dv( &p1[0] );
        glVertex3dv( &p2[0] );
    }
    glEnd();
    /**/
}


}// end of namespace

#include "GLMinSpanTree.h"
#include <opencv2/core/core.hpp>
#include <iostream>


using namespace std;
using namespace cv;
using namespace MST;


namespace GLViewer
{

GLMinSpanTree::GLMinSpanTree( const MST::Graph<MST::Edge, cv::Vec3d>& g,
                              const DisjointSet& djset,
                              const cv::Vec3i& s,
                              const int num_highlights )
    : graph( g ), djs(djset), size( s )
{
    vector<int> index_count(djset.get_size(), 0);
    for( int i=0; i<djset.get_size(); i++ )
    {
        int index = djset.find( i );
        index_count[ index ]++;
    }

    /// Pair<index, count>
    vector<std::pair<int,int> > hightlight_set_indeces( num_highlights, std::pair<int,int>(-1,0) );
    for( int i=0; i<(int)index_count.size(); i++ )
    {
        if( index_count[i]==0 ) continue;
        cout << i << "\t" << index_count[i] << endl;

        int j = 0;
        for( ; j<(int)hightlight_set_indeces.size(); j++ )
        {
            if( index_count[i] < hightlight_set_indeces[j].second ) break;

            if( j>0 )
            {
                hightlight_set_indeces[j-1] = hightlight_set_indeces[j];
            }
        }

        if( j>0 )
        {
            hightlight_set_indeces[j-1].first  = i;
            hightlight_set_indeces[j-1].second = index_count[i];
        }
    }

    cout << endl;
    for( int i=0; i<num_highlights; i++ )
    {
        cout << hightlight_set_indeces[i].first  << "\t " << hightlight_set_indeces[i].second << endl;
    }


    Vec3b default_color[3] =
    {
        Vec3b(255, 0, 0),
        Vec3b(0, 255, 0),
        Vec3b(0, 0, 255)
    };

    for( int i=num_highlights-1; i>=0; i-- )
    {
        Vec3b random_color;
        int default_color_index = num_highlights-1-i;
        if( default_color_index <3 )
        {
            random_color = default_color[ default_color_index ];
        }
        else
        {
            random_color = Vec3b( rand()%255, rand()%255, rand()%255 );
        }
        colors.insert( std::pair<int, Vec3b>( hightlight_set_indeces[i].first, random_color) );
    }
}



GLMinSpanTree::~GLMinSpanTree()
{

}



void GLMinSpanTree::render( void )
{
    glBegin( GL_LINES );
    for( unsigned i=0; i<graph.num_edges(); i++ )
    {
        const Edge& e = graph.get_edge( i );
        const Vec3d& p1 = graph.get_node( e.node1 );
        const Vec3d& p2 = graph.get_node( e.node2 );

        int index = djs.find( e.node1 );
        std::unordered_map<int, Vec3b>::const_iterator it = colors.find( index );
        if( it==colors.end() )
        {
            glColor3ub(50,50,50);
        }
        else
        {
            glColor3ubv( &it->second[0] );
        }

        glVertex3dv( &p1[0] );
        glVertex3dv( &p2[0] );

    }

    glEnd();

    /**/
}


}// end of namespace

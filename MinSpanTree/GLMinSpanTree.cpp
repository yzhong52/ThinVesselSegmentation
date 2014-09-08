#include "GLMinSpanTree.h"
#include <opencv2/core/core.hpp>
#include <iostream>


using namespace std;
using namespace cv;
using namespace MST;


namespace GLViewer
{



GLMinSpanTree::GLMinSpanTree( const MST::Graph<MST::EdgeExt, cv::Vec3d>& g,
                              const DisjointSet& djset,
                              const cv::Vec3i& s,
                              const int num_highlights )
    : graph( g ), djs(djset), size( s ), branch_bit( 1 )
{
    // Count the number of elements in each set
    vector<int> count(djset.get_size(), 0);
    for( int i=0; i<djset.get_size(); i++ )
    {
        int index = djset.find( i );
        count[ index ]++;
    }

    /* Sort 'count' and keep track of 'num_highlights' of the sets with
       the most number of elements.
        - index_count.first: index of a set
        - index_count.second: number of elements in the set */
    vector<std::pair<int,int> > index_count( num_highlights, std::pair<int,int>(-1,0) );
    for( int i=0; i<(int)count.size(); i++ )
    {
        if( count[i]==0 ) continue;

        int j = 0;
        for( ; j<(int)index_count.size(); j++ )
        {
            if( count[i] < index_count[j].second ) break;

            if( j>0 ) index_count[j-1] = index_count[j];
        }

        if( j>0 )
        {
            index_count[j-1].first  = i;
            index_count[j-1].second = count[i];
        }
    }

    // Three default colors to be used
    const Vec3b default_color[3] =
    {
        Vec3b(255,   0,   0), // RED
        Vec3b(  0, 255,   0), // GREEN
        Vec3b(200,  20, 255)  // BLUE-ISH
    };

    for( int i=num_highlights-1; i>=0; i-- )
    {
        Vec3b hight_color;
        const int ranking = num_highlights-1-i;
        if( ranking < 3 )
        {
            // Use default colors
            hight_color = default_color[ ranking ];
        }
        else
        {
            // Generate random colors
            hight_color = Vec3b( rand()%155+100,
                                 rand()%155+100,
                                 rand()%155+100 );
        }
        color_map.insert( std::pair<int, MapElement>(
                              index_count[i].first,
                              MapElement( ranking, hight_color) ) );
    }

    // Find the minimum and maximum sigma of the edges in the graph
    sigma_min = sigma_max = graph.get_edge(0).getSigma();
    for( unsigned i=1; i<graph.num_edges(); i++ )
    {
        const float s = graph.get_edge(i).getSigma();
        sigma_min = std::min( sigma_min, s );
        sigma_max = std::max( sigma_max, s );
    }
    sigma_range = sigma_max - sigma_min;
}



GLMinSpanTree::~GLMinSpanTree()
{

}

void GLMinSpanTree::keyboard( unsigned char key )
{
    /*For example, if color_map.size()==3, then 'branch_bit' will change
      in the folloring order:
       - 0001
       - 0010
       - 0100
       - 0111 */
    const int temp = 1<<color_map.size();

    if( branch_bit == temp-1 )
    {
        branch_bit = 1;
    }
    else
    {
        branch_bit <<= 1;
        if( branch_bit==temp )
        {
            branch_bit--;
        }
    }
}


void GLMinSpanTree::render( void )
{
    glBegin( GL_LINES );
    for( unsigned i=0; i<graph.num_edges(); i++ )
    {
        const EdgeExt& e = graph.get_edge( i );
        const Vec3d& p1 = graph.get_node( e.node1 );
        const Vec3d& p2 = graph.get_node( e.node2 );

        int index = djs.find( e.node1 );
        std::unordered_map<int, MapElement>::const_iterator it;
        it = color_map.find( index );
        if( it==color_map.end() || !( branch_bit & (1<<it->second.rank) ) )
        {
            glColor3ub(25,25,25);
        }
        else
        {
            float val = e.getSigma() / sigma_max;
            // val = std::max(0.03f, val);
            val = sqrt( val );
            const Vec3b c = it->second.color * val;
            glColor3ubv( &c[0] );
        }

        glVertex3dv( &p1[0] );
        glVertex3dv( &p2[0] );

    }

    glEnd();

    /**/
}


}// end of namespace

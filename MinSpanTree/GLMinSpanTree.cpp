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
                              const int& min_num_nodes,
                              const int& max_num_highlights )
    : graph( g ), djs(djset), size( s ), branch_display( 0 )
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
        - count_index.first:  number of elements in the set
        - count_index.second: index of a set */
    int num_highlights = 0;
    vector<std::pair<int,int> > count_index;
    for( int i=0; i<(int)count.size(); i++ )
    {
        if( count[i]!=0 ) {
            count_index.push_back( std::pair<int,int>(count[i], i) );
            if( count[i]>=min_num_nodes ) num_highlights++;
        }
    }
    num_highlights = std::min( num_highlights, max_num_highlights );
    num_highlights = std::max( num_highlights, 5 );

    std::sort( count_index.begin(), count_index.end() );
    std::reverse( count_index.begin(), count_index.end() );

    cout << endl << endl;
    for( int i=num_highlights-1; i>=0; i-- ){
        cout << count_index[i].first << "\t";
        //cout << count_index[i].second << endl;
    }
    cout << endl;
    cout << "Number of branches to hightlight: " << num_highlights << endl;
    cout << "Number of branches in total:      " << count_index.size() << endl;



    // Three default colors to be used
    const Vec3b default_color[3] =
    {
        Vec3b(255,   0,   0), // RED
        Vec3b(  0, 255,   0), // GREEN
        Vec3b(220,  20, 255)  // BLUE-ISH
    };

    for( int i=0; i<num_highlights; i++ )
    {
        Vec3b hight_color;
        const int ranking = i;
        if( ranking < 3 )
        {
            // Use default colors
            hight_color = default_color[ ranking ];
        }
        else
        {
            // Generate color for the others
            hight_color = Vec3b( 0, 0, 255 );
            /*hight_color = Vec3b( rand()%155+100,
                                 rand()%155+100,
                                 rand()%155+100 ); */

        }
        color_map.insert( std::pair<int, MapElement>(
                              count_index[i].second,
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
    /*For example, if color_map.size()==8, then 'branch_display' will change
      in the folloring order:
       0,3,4,5,6,7,0,3,4,5, ... */

    if( branch_display < 3 )
    {
        branch_display = 3;
    }
    else
    {
        branch_display = (branch_display + 1) % color_map.size();
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
        if( it==color_map.end() )
        {
            glColor3ub(25,25,25);
        } else if ( it->second.rank>=3 && branch_display!=it->second.rank ) {
            glColor3ub(25,25,25);
        }
        else if( it->second.rank<3 )
        {
            float val = e.getSigma() / sigma_max;
            val = sqrt( val );
            const Vec3b c = it->second.color * val;
            glColor3ubv( &c[0] );
        } else {
            float val = e.getSigma() / sigma_max;
            val = sqrt( sqrt( val ) );
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

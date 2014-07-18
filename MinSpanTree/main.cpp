#include <iostream>
#include "MSTGraph.h"
#include "MSTEdge.h"
#include "../ModelFitting/ModelSet.h"
#include "../ModelFitting/GLViwerModel.h"

using namespace std;
using namespace MST;

/*Note: A lot of code from ModelFitting is being recompiled and reused here. */

int main()
{
    Mat temp = Mat(200, 200, CV_8UC3);
    cv::imshow( "", temp );

    ModelSet models;
    bool flag = models.deserialize( "data15_134_113_116" );
    if( !flag ) {
        return 0;
    }

    GLViwerModel vis;
    GLViewer::GLLineModel *model = new GLViewer::GLLineModel( Vec3i(134,113,116) );
    model->updatePoints( models.tildaP );
    model->updateModel( models.lines, models.labelID );
    vis.objs.push_back( model );
    vis.display();

    return 0;
}


int example( )
{
    // Yuchen: Minimum Spanning Tree

    /* Test Case Input
     [1] --3-- [2]
      |       / | \
      |     /   |   \
      7    2    4    6
      |  /      |      \
      |/        |        \
     [3] --1-- [4] --5-- [0]

     Result Output
     [1] --3-- [2]
              /
            /
           2
         /
       /
     [3] --1-- [4] --5-- [0] */

    //// Build Graph
    Graph<Edge> graph( 5 );
    graph.add_edge( Edge(1, 2, 3) );
    graph.add_edge( Edge(1, 3, 7) );
    graph.add_edge( Edge(2, 3, 2) );
    graph.add_edge( Edge(2, 4, 4) );
    graph.add_edge( Edge(2, 0, 6) );
    graph.add_edge( Edge(3, 4, 1) );
    graph.add_edge( Edge(4, 0, 5) );

    // Compute Minimum Spanning Tree
    Graph<Edge> tree( graph.num_nodes() );
    graph.get_min_span_tree( tree );

    //// Print Result
    cout << "Original Graph: " << graph << endl;
    cout << "Output MST Tree: "<< tree << endl;

    return true;
}

#include <iostream>
#include <vector>
#include "MSTGraph.h"
#include "MSTEdge.h"
#include "GLMinSpanTree.h"

#include "../ModelFitting/ModelSet.h"
#include "../ModelFitting/GLViwerModel.h"
#include "../ModelFitting/Neighbour26.h"

using namespace std;
using namespace MST;
using namespace cv;

/*Note: A lot of code from ModelFitting is being recompiled and reused here. */


void tree_from_dense_graph( const ModelSet& models, Graph<Edge, cv::Vec3d>& tree )
{

    Graph<Edge, cv::Vec3d> graph;
    // compute the projection point add it to graph
    for( unsigned i=0; i<models.tildaP.size(); i++ )
    {
        const int& lineid1 = models.labelID[i];
        const Line3D* line1   = models.lines[lineid1];
        const cv::Vec3i& pos1 = models.tildaP[i];
        const Vec3d proj = line1->projection( pos1 );
        graph.add_node( proj );
    }

    // connect each pair of the nodes
    for( unsigned i=0; i<graph.num_nodes(); i++ )
    {
        for( unsigned j=i+1; j<graph.num_nodes(); j++ )
        {
            const Vec3d& proj1 = graph.get_node( i );
            const Vec3d& proj2 = graph.get_node( j );
            const Vec3d direction = proj1 - proj2;
            const double dist = sqrt( direction.dot( direction ) );
            graph.add_edge( Edge(i, j, dist) );
        }
    }

    graph.get_min_span_tree( tree );
}

int example();

void tree_from_neighborhood( const ModelSet& models, Graph<Edge, cv::Vec3d>& tree )
{

    Graph<Edge, cv::Vec3d> graph;
    // compute the projection point add it to graph
    for( unsigned i=0; i<models.tildaP.size(); i++ )
    {
        const int& lineid1 = models.labelID[i];
        const Line3D* line1   = models.lines[lineid1];
        const cv::Vec3i& pos1 = models.tildaP[i];
        const Vec3d proj = line1->projection( pos1 );
        graph.add_node( proj );
    }

    for( unsigned i=0; i<models.tildaP.size(); i++ )
    {
        for( int n=0; n<26; n++ )
        {
            const int& lineid1 = models.labelID[i];
            const Line3D* line1   = models.lines[lineid1];
            const cv::Vec3i& pos1 = models.tildaP[i];

            cv::Vec3i pos2;
            Neighbour26::getNeigbour( n, pos1, pos2 );
            if ( !models.labelID3d.isValid( pos2 ) ) continue;

            const int lineid2 = models.labelID3d.at( pos2 );
            if ( lineid2==-1 ) continue;

            const Line3D* line2   = models.lines[lineid2];

            const Vec3d proj1 = line1->projection( pos1 );
            const Vec3d proj2 = line2->projection( pos2 );
            const Vec3d direction = proj1 - proj2;
            const double dist = sqrt( direction.dot( direction ) );

            // TODO: line id is equivalent to point id in this case
            graph.add_edge( Edge(lineid1, lineid2, dist) );
        }
    }

    graph.get_min_span_tree( tree );
}

int main()
{
    Mat temp = Mat(200, 200, CV_8UC3);
    cv::imshow( "", temp );

    ModelSet models;
    bool flag = models.deserialize( "vessel3d_rd_585_525_300" );
    if( !flag )
        return 0;

    GLViwerModel vis;

    //*
    Graph<Edge, cv::Vec3d> tree1;
    tree_from_neighborhood( models, tree1 );
    GLViewer::GLMinSpanTree *mstobj1 = new GLViewer::GLMinSpanTree( tree1, models.labelID3d.get_size() );
    mstobj1->set_color( Vec3f(1.0f, 1.0f, 0.0f) );
    vis.objs.push_back( mstobj1 );

    /*
    Graph<Edge, cv::Vec3d> tree2;
    tree_from_dense_graph( models, tree2 );
    GLViewer::GLMinSpanTree *mstobj2 = new GLViewer::GLMinSpanTree( tree2, models.labelID3d.get_size() );
    vis.objs.push_back( mstobj2 );
    mstobj2->set_color( Vec3f(0.0f, 0.0f, 1.0f) );
    /**/

    //*
    GLViewer::GLLineModel *model = new GLViewer::GLLineModel( models.labelID3d.get_size() );
    model->updatePoints( models.tildaP );
    model->updateModel( models.lines, models.labelID );
    vis.objs.push_back( model );
    /**/


    Data3D<short> im_short;
    flag = im_short.load( "../temp/vessel3d_rd.data", Vec3i(585, 525, 892), false, true );
    // flag = im_short.load( "../data/data15.data" );
    if( !flag ) return 0;
    // im_short.remove_margin_to( Vec3i(585, 525, 10) );
    im_short.remove_margin_to( Vec3i(585, 525, 300) );
    im_short.remove_margin( Vec3i(0,0,0), Vec3i(385, 325, 200) );
    vis.addObject( im_short,  GLViewer::Volumn::MIP );

    vis.display(640, 480, 3);

    return 0;
}



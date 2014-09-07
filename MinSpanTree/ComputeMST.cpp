#include "ComputeMST.h"

#include <iostream>
#include <iomanip>      // std::setprecision

#include "MSTEdgeExt.h"
#include "DisjointSet.h"
#include "../ModelFitting/Line3DTwoPoint.h"
#include "../ModelFitting/Neighbour26.h"

using namespace MST;
using namespace cv;
using namespace std;

double (*ComputeMST::edge_weight_func)( const Line3D*, const cv::Vec3d&,
                                        const Line3D*, const cv::Vec3d& )
    = ComputeMST::edge_weight_func_distance;

void ComputeMST::from_threshold_graph( const ModelSet& models,
                                       const Data3D<unsigned char>& mask,
                                       Graph<EdgeExt, cv::Vec3d>& tree,
                                       DisjointSet& djs )
{
    cout << "Compute MST Begin (from_threshold_graph)" << endl;

    Graph<EdgeExt, cv::Vec3d> graph;

    /// compute the projection point add it to graph
    create_graph_nodes( models, graph );

    /// connect each pair of the nodes with a threshold
    int counter = 0;
    const int total = graph.num_nodes();
    #pragma omp parallel for schedule(dynamic)
    for( unsigned i=0; i<graph.num_nodes(); i++ )
    {
        #pragma omp critical
        {
            long long temp = (long long)(1000) * ++counter / total;
            cout << "\r ";
            cout.width(10);
            cout << fixed << setprecision(1) << temp / 10.0 << "%    ";
            cout.flush();
        }

        for( unsigned j=i+1; j<graph.num_nodes(); j++ )
        {
            const int& lineidi  = models.labelID[i];
            const Line3D* linei = models.lines[lineidi];


            const int& lineidj  = models.labelID[j];
            const Line3D* linej = models.lines[lineidj];

            const Vec3d& proj1 = graph.get_node( i );
            const Vec3d& proj2 = graph.get_node( j );


            const double dist = edge_weight_func( linei, proj1, linej, proj2 );

            /// Discard if the distance between the two projection points are greater than
            /// twice the sum of the vessel size
            const double threshold = 2.0 * (linei->getSigma() + linej->getSigma());
            if( dist>threshold ) continue;

            /// Weight is between [0.5-1] after this manipulation
            double weight = 1.0;
            /*
            const Vec3d& directioni = linei->getDirection();
            const Vec3d& directionj = linej->getDirection();
            const Vec3d norm_direction = direction / sqrt( direction.dot(direction) );
            weight  = abs( norm_direction.dot( directioni ) );
            weight += abs( norm_direction.dot( directionj ) );
            weight = (2 - weight) / 4;
            weight = sqrt( weight );
            /**/

            #pragma omp critical
            {
                graph.add_edge( EdgeExt(i, j,
                                     dist*weight,
                                     std::min(linei->getSigma(), linej->getSigma()) ) );
            }
        }
    }

    graph.get_min_span_tree( tree, &djs );
    cout << endl << "Done" << endl << endl;
}


void ComputeMST::neighborhood_graph( const ModelSet& models,
                                     const Data3D<unsigned char>& mask,
                                     MST::Graph<MST::EdgeExt, cv::Vec3d>& tree,
                                     DisjointSet& djs )
{
    cout << "Compute MST Begin (from neighborhood graph)..." << endl;

    Graph<EdgeExt, cv::Vec3d> graph;
    create_graph_nodes( models, graph );

    for( unsigned i=0; i<models.tildaP.size(); i++ )
    {
        for( int n=0; n<26; n++ )
        {
            // Current position
            const cv::Vec3i& pos1 = models.tildaP[i];

            // Neibhgour position
            cv::Vec3i pos2 = Neighbour26::at( n ) + pos1;
            if ( !models.pointID3d.isValid( pos2 ) ) continue;

            const int j = models.pointID3d.at( pos2 );
            if ( j==-1 ) continue;

            const int& lineid1 = models.labelID[ i ];
            const int& lineid2 = models.labelID[ j ];

            const Line3D* line1 = models.lines[lineid1];
            const Line3D* line2 = models.lines[lineid2];
            const Vec3d& proj1 = graph.get_node( i );
            const Vec3d& proj2 = graph.get_node( j );

            const double dist = edge_weight_func( line1,proj1,line2,proj2 );

            /* Discard if the distance between the two projection points are
               greater than twice the sum of the vessel size */
            const double threshold = 2.0 * (line1->getSigma() + line2->getSigma());
            if( dist>threshold ) continue;

            graph.add_edge( EdgeExt( i, j, dist,
                                  std::min(line1->getSigma(), line2->getSigma()) ) );
        }
    }

    graph.get_min_span_tree( tree, &djs );


    // Determine critical points which are connected to at most one
    // other points
    vector<unsigned> critical_points;
    vector<unsigned> neighbor_counts( tree.num_nodes(), 0 );
    for( unsigned int i=0; i<tree.num_edges(); i++ )
    {
        const Edge &e = tree.get_edge( i );
        neighbor_counts[ e.node1 ]++;
        neighbor_counts[ e.node2 ]++;
    }

    for( unsigned i=0; i<neighbor_counts.size(); i++ )
    {
        if( neighbor_counts[i]==1 )
        {
            critical_points.push_back( i );
        }
    }

    cout << "\t Number of critical points: ";
    cout << critical_points.size() << endl;
    cout << "\t Number of nodes in the graph: ";
    cout << graph.num_nodes() << endl;

    // Add more edges to the graph based on the tree
    graph = tree;

    int counter = 0;
    const int total = critical_points.size();
    #pragma omp parallel for schedule(dynamic)
    for( unsigned k=0; k<critical_points.size(); k++ )
    {
        #pragma omp critical
        {
            long long temp = (long long)(1000) * ++counter / total;
            cout << "\r ";
            cout.width(10);
            cout << fixed << setprecision(1) << temp / 10.0 << "%    ";
            cout.flush();
        }

        const unsigned i = critical_points[k];

        const int& lineidi  = models.labelID[i];
        const Line3D* linei = models.lines[lineidi];
        const Vec3d& proj1 = graph.get_node( i );

        for( unsigned j=0; j<graph.num_nodes(); j++ )
        {
            if( i==j ) continue;
            if( i<j && neighbor_counts[j]==1 ) continue;

            const int& lineidj  = models.labelID[j];
            const Line3D* linej = models.lines[lineidj];
            const Vec3d& proj2 = graph.get_node( j );

            const double dist = edge_weight_func( linei, proj1, linej, proj2 );

            /* Discard if the distance between the two projection points are
               greater than twice the sum of the vessel size */
            const double threshold = 2.0 * (linei->getSigma() + linej->getSigma());
            if( dist>threshold ) continue;

            #pragma omp critical
            {
                graph.add_edge( EdgeExt(i, j, dist,
                                        std::min(linei->getSigma(), linej->getSigma()) ) );
            }
        }
    }

    /// build edges
    std::priority_queue<EdgeExt> edges = graph.get_edges();
    while( !edges.empty() && tree.num_edges()<tree.num_nodes()-1 )
    {
        EdgeExt e = edges.top();
        const int sid1 = djs.find( e.node1 );
        const int sid2 = djs.find( e.node2 );
        if( sid1 != sid2 )
        {
            tree.add_edge( e );
            djs.merge( sid1, sid2 );
        }
        edges.pop();
    }
}

void ComputeMST::create_graph_nodes( const ModelSet& models,
                                     Graph<EdgeExt, cv::Vec3d>& graph )
{
    // compute the projection point add it to graph
    for( unsigned i=0; i<models.tildaP.size(); i++ )
    {
        const int& lineid1 = models.labelID[i];
        const Line3D* line1   = models.lines[lineid1];
        const cv::Vec3i& pos1 = models.tildaP[i];
        const Vec3d proj = line1->projection( pos1 );
        graph.add_node( proj );
    }
}


double ComputeMST::edge_weight_func_distance( const Line3D* line1,
        const cv::Vec3d& proj1,
        const Line3D* line2,
        const cv::Vec3d& proj2 )
{
    const Vec3d direction = proj1 - proj2;

    double dist = sqrt( direction.dot( direction ) );

    // The following would make the result looks worse!
    // dist -= max(line1->getSigma(), line2->getSigma());
    // dist = max( dist, 0.0 );

    return dist;
}

double edge_weight_func_distance_and_direction( const Line3D* line1,
        const cv::Vec3i& pos1,
        const Line3D* line2,
        const cv::Vec3i& pos2)
{
    const Vec3d proj1 = line1->projection( pos1 );
    const Vec3d proj2 = line2->projection( pos2 );
    const Vec3d direction = proj1 - proj2;

    double dist = sqrt( direction.dot( direction ) );
    return max( 0.0, dist - max(line1->getSigma(),line2->getSigma() ) );
}


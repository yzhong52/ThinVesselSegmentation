#include "ComputeMST.h"

#include <iostream>
#include <iomanip>      // std::setprecision

#include "MSTEdge.h"
#include "DisjointSet.h"
#include "../ModelFitting/Line3DTwoPoint.h"

using namespace MST;
using namespace cv;
using namespace std;

void ComputeMST::from_threshold_graph( const ModelSet& models,
                                      Graph<Edge, cv::Vec3d>& tree,
                                      DisjointSet& djs )
{
    cout << "Compute MST Begin (from_threshold_graph)" << endl;

    Graph<Edge, cv::Vec3d> graph;

    /// compute the projection point řadd it to graph
    for( unsigned i=0; i<models.tildaP.size(); i++ )
    {
        const int& lineid1 = models.labelID[i];
        const Line3D* line1   = models.lines[lineid1];
        const cv::Vec3i& pos1 = models.tildaP[i];
        const Vec3d proj = line1->projection( pos1 );
        graph.add_node( proj );
    }

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
            const Vec3d& proj1 = graph.get_node( i );
            const Vec3d& proj2 = graph.get_node( j );

            // TODO: REMOVE
            /*
            if( proj1[2] < models.labelID3d.SZ()/2 ) continue;
            if( proj2[2] < models.labelID3d.SZ()/2 ) continue;
            /**/

            const int& lineidi  = models.labelID[i];
            const Line3D* linei = models.lines[lineidi];
            const int& lineidj  = models.labelID[j];
            const Line3D* linej = models.lines[lineidj];

            const Vec3d direction = proj1 - proj2;
            double dist  = sqrt( direction.dot( direction ) );
            dist -= max( linei->getSigma(), linej->getSigma() );
            dist = max( 0.0, dist );

            /// Discard if the distance between the two projection points are greater than
            /// twice the sum of the vessel size
            const double threshold = 1.0 * (linei->getSigma() + linej->getSigma());
            if( dist>threshold ) continue;

            /// Weight is between [0.5-1] after this manipulation
            double weight = 1.0;
            /*
            const Vec3d& directioni = linei->getDirection();
            const Vec3d& directionj = linej->getDirection();
            const Vec3d norm_direction = direction / sqrt( direction.dot(direction) );
            weight += abs( norm_direction.dot( directioni ) );
            weight += abs( norm_direction.dot( directionj ) );
            weight = (2 - weight) / 4;
            weight = sqrt( weight );
            */

            #pragma omp critical
            {
                graph.add_edge( Edge(i, j, dist*weight ) );
            }
        }
    }

    graph.get_min_span_tree( tree, &djs );
    cout << endl << "Done" << endl << endl;
}


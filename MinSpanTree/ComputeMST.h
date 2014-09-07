#ifndef COMPUTEMST_H
#define COMPUTEMST_H

#include "MSTGraph.h"
#include "../ModelFitting/ModelSet.h"

namespace MST
{
class EdgeExt;
template<class EdgeType, class NodeType> class Graph;
}

class ComputeMST
{
public:
/// 1. weighted distance based on direction
/// 2. Taken each pair of nodes into consideration
/// 3. Discard a pair if the distance between them is greater than a certain threshold.
///    The threshold is computed automatically based on the thickness of the vessels.
    static void from_threshold_graph( const ModelSet& models,
                                      const Data3D<unsigned char>& mask,
                                      MST::Graph<MST::EdgeExt, cv::Vec3d>& tree,
                                      DisjointSet& djs );

    static void neighborhood_graph( const ModelSet& models,
                                    const Data3D<unsigned char>& mask,
                                    MST::Graph<MST::EdgeExt, cv::Vec3d>& tree,
                                    DisjointSet& djs );

    // TODO: different options of computing distance edge weights
    enum Option
    {
        DISTANCE_ONLY,
        DISTANCE_WITH_HEURISTIC_ANGLE
    };
    inline static void set_edge_weight_func( Option option )
    {
        switch( option )
        {
        case DISTANCE_ONLY:
            edge_weight_func = edge_weight_func_distance;
            break;
        case DISTANCE_WITH_HEURISTIC_ANGLE:
            edge_weight_func = edge_weight_func_distance_and_direction;
            break;
        }
    }

private:
    static double (*edge_weight_func)( const Line3D*, const cv::Vec3d&,
                                       const Line3D*, const cv::Vec3d& );

    static double edge_weight_func_distance( const Line3D* line1,
            const cv::Vec3d& proj1,
            const Line3D* line2,
            const cv::Vec3d& proj2 );
    static double edge_weight_func_distance_and_direction( const Line3D* line1,
            const cv::Vec3d& proj1,
            const Line3D* line2,
            const cv::Vec3d& proj2);

    static void create_graph_nodes( const ModelSet& models,
                                    MST::Graph<MST::EdgeExt, cv::Vec3d>& graph );

    static void add_graph_edge( const ModelSet& models,
                                MST::Graph<MST::EdgeExt, cv::Vec3d>& graph,
                                const int& index1, const int index2 );
};

#endif // COMPUTEMST_H

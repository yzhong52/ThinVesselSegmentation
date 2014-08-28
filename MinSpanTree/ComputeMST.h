#ifndef BUILDTREE_H
#define BUILDTREE_H

#include "MSTGraph.h"
#include "../ModelFitting/ModelSet.h"

namespace ComputeMST
{
/// 1. weighted distance based on direction
/// 2. Taken each pair of nodes into consideration
/// 3. Discard a pair if the distance between them is greater than a certain threshold.
///    The threshold is computed automatically based on the thickness of the vessels.
void from_threshold_graph( const ModelSet& models,
                           MST::Graph<MST::Edge, cv::Vec3d>& tree,
                           DisjointSet& djs );
}

#endif // BUILDTREE_H

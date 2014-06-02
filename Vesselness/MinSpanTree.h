#pragma once

#include <iostream>

#include "Graph.h"
#include "Edge.h"
#include "DataTypes.h"

namespace MinSpanTree
{
// A naive example
inline int build_tree_example();

// pre-processing xuefeng's data (rings reduction)
bool pre_process_xuefeng( const std::string& file_name,
                          const std::string& save_file_name,
                          /*Output*/ Graph<Edge_Ext, LineSegment>& ring,
                          /*INPUT*/  Vec3f center_of_ring );

// build_minimum_spanning_tree
bool build_tree_xuefeng( const std::string& file_name,
                         /*Output*/ Graph<Edge_Ext, LineSegment>& tree,
                         /*Input*/ float thres = 0.0f );
}

namespace MST = MinSpanTree;

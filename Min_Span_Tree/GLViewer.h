#pragma once

#include "Edge_Graph.h"
#include <vector>

class Edge_Ext;

namespace GLViewer
{
	void show_dir( const Edge_Graph<Edge_Ext>& tree, const std::vector<Line>& lines );
};


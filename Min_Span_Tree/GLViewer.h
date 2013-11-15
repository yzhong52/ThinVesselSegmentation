#pragma once

#include "Edge_Graph.h"
#include "Edge.h"
#include <vector>
using namespace std;

namespace GLViewer
{
	void MIP( unsigned char* data, int x, int y, int z, Edge_Graph<Edge_Ext>& tree, vector<Line>& lines );
}


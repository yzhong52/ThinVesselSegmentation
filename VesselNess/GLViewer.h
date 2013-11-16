#pragma once

#include "Graph.h"
#include "Edge.h"
#include "DataTypes.h"

#include <vector>
using namespace std;

namespace GLViewer
{
	// pre_draw_func: can be used to add additional object to render
	// post_draw_func: can be used to save the framebuffer
	void MIP( unsigned char* data, int x, int y, int z,
		void (*pre_draw_func)(void) = NULL, 
		void (*post_draw_func)(int, int) = NULL);
}


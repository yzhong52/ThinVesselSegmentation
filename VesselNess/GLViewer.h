#pragma once

#include "Graph.h"
#include "Edge.h"
#include "DataTypes.h"

#include <vector>
using namespace std;



namespace GLViewer
{
	class Object{
	public:
		// init function for OpenGL
		virtual void init(void) = 0;
		// render function for OpenGL
		virtual void render(void) = 0;
		// size of the object
		virtual unsigned int size_x(void) const = 0;
		virtual unsigned int size_y(void) const = 0;
		virtual unsigned int size_z(void) const = 0;
	}; 

	void MIP( unsigned char* data, int x, int y, int z,
		void (*pre_draw_func)(void) = NULL, 
		void (*post_draw_func)(int, int) = NULL );

	// pre_draw_func: can be used to add additional object to render
	// post_draw_func: can be used to save the framebuffer
	void MIP2( vector<Object*> objects ); 

	// TODO: to visualized color texture
	void MIP_color( unsigned char* data, int x, int y, int z,
		void (*pre_draw_func)(void) = NULL, 
		void (*post_draw_func)(int, int) = NULL);
}


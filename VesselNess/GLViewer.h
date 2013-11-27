#pragma once

#include <vector>
using namespace std;

namespace GLViewer
{
	class Object{
	public:
		// init function for OpenGL
		virtual void init(void) { };
		// render function for OpenGL
		virtual void render(void) = 0;
		// size of the object
		virtual unsigned int size_x(void) const = 0;
		virtual unsigned int size_y(void) const = 0;
		virtual unsigned int size_z(void) const = 0;
	}; 

	// pre_draw_func: can be used to add additional object to render
	// post_draw_func: can be used to save the framebuffer
	void go( vector<Object*> objects ); 

	// TODO: to visualized color texture
}


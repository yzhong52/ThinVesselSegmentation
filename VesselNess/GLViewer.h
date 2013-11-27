#pragma once

#include <vector>
using namespace std;

#include "VideoSaver.h"

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

	void go( vector<Object*> objects, VideoSaver* videoSaver = NULL ); 

	// TODO: to visualized color texture
}


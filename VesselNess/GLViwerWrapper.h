#pragma once

#include "GLViewer.h"

// Using OpenCV ot Save the Scene to video
#define _CRT_SECURE_NO_DEPRECATE
#include <opencv\cv.h>      
#include "Data3D.h"

namespace GLViewer{
	void MIP( Data3D<unsigned char>& im_uchar, 
		void (*pre_draw_func)(void) = NULL, 
		void (*post_draw_func)(int, int) = NULL)
	{
		MIP( im_uchar.getMat().data, 
			im_uchar.SX(), im_uchar.SY(), im_uchar.SZ(), 
			pre_draw_func, post_draw_func );
	}
}
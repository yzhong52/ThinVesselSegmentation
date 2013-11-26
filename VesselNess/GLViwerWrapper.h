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

	void MIP( Data3D<short>& im_short, 
		void (*pre_draw_func)(void) = NULL, 
		void (*post_draw_func)(int, int) = NULL)
	{
		// normalize the data
		IP::normalize( im_short, short(255) );
		// convert to unsigned char
		Data3D<unsigned char> im_uchar;
		im_short.convertTo( im_uchar );
		// visualize 
		MIP( im_uchar.getMat().data, 
			im_uchar.SX(), im_uchar.SY(), im_uchar.SZ(), 
			pre_draw_func, post_draw_func );
	}

	void MIP( Data3D<float>& vn_float, 
		void (*pre_draw_func)(void) = NULL, 
		void (*post_draw_func)(int, int) = NULL)
	{
		// normalize the data
		IP::normalize( vn_float, float(255) );
		// convert to unsigned char
		Data3D<unsigned char> im_uchar;
		vn_float.convertTo( im_uchar );
		// visualize 
		MIP( im_uchar, pre_draw_func, post_draw_func );
	}

	template<class T>
	void MIP( Data3D<T>& vesselness, 
		void (*pre_draw_func)(void) = NULL, 
		void (*post_draw_func)(int, int) = NULL)
	{
		// copy the first dimension to vn
		Data3D<float> vn;
		vesselness.copyDimTo( vn, 0 );

		MIP( vn, pre_draw_func, post_draw_func );
	}
}
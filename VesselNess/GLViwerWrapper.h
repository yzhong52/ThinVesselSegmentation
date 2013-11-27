#pragma once

// Using OpenCV ot Save the Scene to video
#define _CRT_SECURE_NO_DEPRECATE
#include <opencv\cv.h>

#include "Data3D.h"
#include "Volumn.h"

namespace GLViewer{
	void MIP( Data3D<unsigned char>& im_uchar )
	{
		GLViewerExt::Volumn vObj( im_uchar.getMat().data, im_uchar.SX(), im_uchar.SY(), im_uchar.SZ() );
		vector<GLViewer::Object*> objs;
		objs.push_back( &vObj );
		GLViewer::go( objs );
	}

	void MIP( Data3D<short>& im_short )
	{
		// normalize the data
		IP::normalize( im_short, short(255) );
		// convert to unsigned char
		Data3D<unsigned char> im_uchar;
		im_short.convertTo( im_uchar );
		// visualize 
		MIP( im_uchar );
	}

	void MIP( Data3D<float>& vn_float )
	{
		// normalize the data
		IP::normalize( vn_float, float(255) );
		// convert to unsigned char
		Data3D<unsigned char> im_uchar;
		vn_float.convertTo( im_uchar );
		// visualize 
		MIP( im_uchar );
	}

	template<class T>
	void MIP( Data3D<T>& vesselness )
	{
		// copy the first dimension to vn
		Data3D<float> vn;
		vesselness.copyDimTo( vn, 0 );
		MIP( vn );
	}
}
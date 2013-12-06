#pragma once

// Using OpenCV ot Save the Scene to video
#define _CRT_SECURE_NO_DEPRECATE
#include <opencv\cv.h>

#include "Data3D.h"
#include "Volumn.h"
#include "MinSpanTreeWrapper.h"
#include "CenterLine.h"

class GLViewerExt{
private:
	vector<GLViewer::Object*> objs;

public:
	~GLViewerExt() {
		for( unsigned int i=0; i<objs.size(); i++ ) {
			delete objs[i];
			objs[i] = NULL; 
		}
	}

	template<class T>
	void addObject( Data3D<T>& im_data )
	{
		// change the data formate to unsigend char
		Data3D<unsigned char> im_uchar;
		IP::normalize( im_data, T(255) );
		im_data.convertTo( im_uchar ); 
		addObject( im_uchar );
	}

	template<>
	void addObject( Data3D<unsigned char>& im_uchar ) {
		// copy the data
		GLViewer::Volumn* vObj = new GLViewer::Volumn( im_uchar.getMat().data,
			im_uchar.SX(), im_uchar.SY(), im_uchar.SZ() );
		objs.push_back( vObj );
	}

	template<>
	void addObject( Data3D<Vesselness_All>& vn_all ) {
		Data3D<float> vn_float;
		vn_all.copyDimTo( vn_float, 0 ); 
		this->addObject( vn_float );
	}
	
	template<>
	void addObject( Data3D<Vesselness_Sig>& vn_sig ) {
		Data3D<float> vn_float;
		vn_sig.copyDimTo( vn_float, 0 ); 
		this->addObject( vn_float );
	}

	template<class E, class N>
	GLViewer::CenterLine<E, N>* addObject( Graph< E, N >& tree ) {
		GLViewer::CenterLine<E, N> *cObj = new GLViewer::CenterLine<E, N>( tree ); 
		objs.push_back( cObj );
		// return the pointer to the rendering object so that
		// we can have addtional configuration to it (if we want to)
		return cObj; 
	}
	
	template<class E, class N>
	void addObject( MinSpanTree::Graph3D< E, N >& tree ) {
		GLViewer::CenterLine<E, N> *cObj = new GLViewer::CenterLine<E, N>( tree ); 
		objs.push_back( cObj );
	}

	void go() {
		GLViewer::go( objs );
	}

	void saveVideo() {
		GLViewer::VideoSaver videoSaver( "output/video.avi" );
		GLViewer::go( objs, &videoSaver );
	}
}; 
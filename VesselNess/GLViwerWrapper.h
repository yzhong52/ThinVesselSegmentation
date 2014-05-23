#pragma once

// Using OpenCV ot Save the Scene to video
#define _CRT_SECURE_NO_DEPRECATE
#include <opencv2\opencv.hpp>

#include "Data3D.h"
#include "GLVolumn.h"
#include "GLDirection.h"
#include "GLCenterLine.h"
#include "GLViewer.h"
#include "MinSpanTreeWrapper.h"

/* Example: Rendering a 3D volume using maximum intensity projection. *//*
    //
    GLViewerExt ver;
    Data3D<short> im_short;
    im_short.load( "data/roi15.data" );
    ver.addObject( im_short, GLViewer::Volumn::MIP );
    // Start Rendering
    ver.go();
*/

class GLViewerExt{
public:
    // a list of objects that to be rendered
    vector<GLViewer::Object*> objs;

public:
    ~GLViewerExt() {
        for( unsigned int i=0; i<objs.size(); i++ ) delete objs[i];
    }

    template<class T>
    void addObject( const Data3D<T>& im_data, GLViewer::Volumn::RenderMode mode = GLViewer::Volumn::MIP )
    {
        if( im_data.get_size_total()==0 ) return;
        // normalized the data
        Data3D<T> im_copy = im_data;
        IP::normalize( im_copy, T(255) );
        // change the data formate to unsigend char
        Data3D<unsigned char> im_uchar;
        im_data.convertTo( im_uchar );
        addObject( im_uchar, mode );
    }





    template<class E, class N>
    GLViewer::CenterLine<E, N>* addObject( MST::Graph< E, N >& tree ) {
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


    void addDiretionObject( Data3D<Vesselness_Sig>& vn_sig ) {
        GLViewer::Direction* vDir = new GLViewer::Direction( vn_sig );
        objs.push_back( vDir );
    }

    void go( int w = 1280, int h = 720, int numViewports = 1 ) {
        GLViewer::numViewports = numViewports;
        GLViewer::go( objs, w, h );
    }
};


    template<>
    void GLViewerExt::addObject( const Data3D<unsigned char>& im_uchar, GLViewer::Volumn::RenderMode mode ) {
        // copy the data
        GLViewer::Volumn* vObj = new GLViewer::Volumn(
            im_uchar.getMat().data,
            im_uchar.SX(),
            im_uchar.SY(),
            im_uchar.SZ(), &GLViewer::cam );
        vObj->render_mode = mode;
        objs.push_back( vObj );
    }

    template<>
    void GLViewerExt::addObject( const Data3D<Vesselness_All>& vn_all, GLViewer::Volumn::RenderMode mode ) {
        Data3D<float> vn_float;
        vn_all.copyDimTo( vn_float, 0 );
        this->addObject( vn_float, mode );
    }

    template<>
    void GLViewerExt::addObject( const Data3D<Vesselness_Sig>& vn_sig, GLViewer::Volumn::RenderMode mode ) {
        Data3D<float> vn_float;
        vn_sig.copyDimTo( vn_float, 0 );
        this->addObject( vn_float, mode );
    }

    template<>
    void GLViewerExt::addObject( const Data3D<Vesselness_Nor>& vn_nor, GLViewer::Volumn::RenderMode mode ) {
        Data3D<float> vn_float;
        vn_nor.copyDimTo( vn_float, 0 );
        this->addObject( vn_float, mode );
    }

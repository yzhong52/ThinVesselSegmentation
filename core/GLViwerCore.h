#ifndef GLVIWERCORE_H
#define GLVIWERCORE_H

#include <vector>
#include "GLViewer.h"
#include "GLVolumn.h"
#include "ImageProcessing.h"

class GLViewerCore
{
public:
    // a list of objects that to be rendered
    std::vector<GLViewer::Object*> objs;
    virtual ~GLViewerCore()
    {
        for( unsigned int i=0; i<objs.size(); i++ )
            delete objs[i];
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

    void display( int w = 1280, int h = 720, int numViewports = 1 )
    {
        GLViewer::numViewports = numViewports;
        GLViewer::dispay( objs, w, h );
    }
};


template<>
void GLViewerCore::addObject( const Data3D<unsigned char>& im_uchar, GLViewer::Volumn::RenderMode mode )
{
    // copy the data
    GLViewer::Volumn* vObj = new GLViewer::Volumn(
        im_uchar.getMat().data,
        im_uchar.SX(),
        im_uchar.SY(),
        im_uchar.SZ(), &GLViewer::camera );
    vObj->render_mode = mode;
    objs.push_back( vObj );
}

#endif // GLVIWERCORE_H

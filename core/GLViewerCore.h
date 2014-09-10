#ifndef GLVIEWERCORE_H
#define GLVIEWERCORE_H

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
        {
            delete objs[i];
        }
    }

    template<class T>
    void addObject( const Data3D<T>& im_data,
                    GLViewer::Volumn::RenderMode mode = GLViewer::Volumn::MIP );

    void display( int w = 1280, int h = 720, int numViewports = 1 )
    {
        GLViewer::numViewports = numViewports;
        GLViewer::dispay( objs, w, h );
    }

    void addUcharObject( const Data3D<unsigned char>& im_data,
                         GLViewer::Volumn::RenderMode mode = GLViewer::Volumn::MIP,
                         float scale = 1.0f );
};


template<class T>
void GLViewerCore::addObject( const Data3D<T>& im_data,
                              GLViewer::Volumn::RenderMode mode )
{
    // data is empty
    if( im_data.is_empty() ) return;

    // normalized the data
    Data3D<T> im_copy = im_data;
    IP::normalize( im_copy, T(255) );

    // change the data formate to unsigend char
    Data3D<unsigned char> im_uchar;
    im_data.convertTo( im_uchar );

    // TODO: Magic number below. In order to increase rendering speed,
    // resize the volume to a smaller size and scale it up while rendering.
    float scale = 1.0f;
    if( im_uchar.SZ()>400 )
    {
        im_uchar.shrink_by_half();
        scale = 2.0f;
    }

    this->addUcharObject( im_uchar, mode, scale );
}

#endif // GLVIEWERCORE_H

#include "GLViewerCore.h"

void GLViewerCore::addUcharObject( const Data3D<unsigned char>& im_uchar,
                                   GLViewer::Volumn::RenderMode mode,
                                   float scale )
{
    // copy the data from 'Data3D' to 'Volume'.
    GLViewer::Volumn* vObj = new GLViewer::Volumn(
        im_uchar.getMat().data,
        im_uchar.SX(),
        im_uchar.SY(),
        im_uchar.SZ(), &GLViewer::camera, scale );

    vObj->render_mode = mode;
    objs.push_back( vObj );
}

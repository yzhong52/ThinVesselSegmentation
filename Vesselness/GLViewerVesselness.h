#ifndef GLVIWERVESSELNESS_H
#define GLVIWERVESSELNESS_H

#include "GLViwerCore.h"
#include "GLDirection.h"

class GLViwerVesselness : public GLViewerCore {
public:
    void addDiretionObject( Data3D<Vesselness_Sig>& vn_sig )
    {
        GLViewer::Direction* vDir = new GLViewer::Direction( vn_sig );
        objs.push_back( vDir );
    }
};

template<>
void GLViewerCore::addObject( const Data3D<Vesselness_All>& vn_all, GLViewer::Volumn::RenderMode mode )
{
    Data3D<float> vn_float;
    vn_all.copyDimTo( vn_float, 0 );
    this->addObject( vn_float, mode );
}

template<>
void GLViewerCore::addObject( const Data3D<Vesselness_Sig>& vn_sig, GLViewer::Volumn::RenderMode mode )
{
    Data3D<float> vn_float;
    vn_sig.copyDimTo( vn_float, 0 );
    this->addObject( vn_float, mode );
}

template<>
void GLViewerCore::addObject( const Data3D<Vesselness_Nor>& vn_nor, GLViewer::Volumn::RenderMode mode )
{
    Data3D<float> vn_float;
    vn_nor.copyDimTo( vn_float, 0 );
    this->addObject( vn_float, mode );
}

#endif

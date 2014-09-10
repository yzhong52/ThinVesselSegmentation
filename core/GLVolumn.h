#ifndef GL_VOLUMN_H
#define GL_VOLUMN_H


#ifndef _CRT_SECURE_NO_DEPRECATE
#define _CRT_SECURE_NO_DEPRECATE
#endif
#include <queue>
#include <new> // for no throw

//#include "Graph.h"
//#include "MinSpanTree.h"
//#include "MinSpanTreeWrapper.h"
//#include "DataTypes.h"

#include <opencv2/core/core.hpp>
#include <iostream>

#include "GLViewer.h"
#include "GLCamera.h"

namespace GLViewer
{
// rendering object with Maximum Intenstiy Projection
class Volumn : public GLViewer::Object
{
public:
    // rendeing mode
    enum RenderMode
    {
        MIP, //Maximum Intensity Projection
        CrossSection,
        Surface
    } render_mode;

private:
    /////////////////////////////////////////
    // Data
    ///////////////////////
    // 3D Volumn Texture
    GLuint texture;
    // size of texture
    int texture_sx, texture_sy, texture_sz;
    // Original Data
    int sx, sy, sz;
    // Texture Data
    unsigned char* data;
    // Reference to the camera
    GLCamera* ptrCam;

    float scale;

public:
    Volumn(unsigned char* im_data,
           const int& im_x, const int& im_y, const int& im_z,
           GLCamera* ptrCamera,
           float scale = 1.0f );

    ~Volumn();

    bool update_data( unsigned char* im_data );

    void init(void);

    void setRenderMode( RenderMode mode );

    virtual void keyboard( unsigned char key )
    {
        if( key!='\t' ) return;

        switch (render_mode)
        {
        case MIP:
            setRenderMode( CrossSection );
            break;
        case CrossSection:
            setRenderMode( Surface );
            break;
        case Surface:
            setRenderMode( MIP );
            break;
        }
    }

    std::vector<cv::Vec3f> intersectPoints( const cv::Vec3f& center, const cv::Vec3f& norm );


    void render_volumn( const float& dx = 1.0f,
                        const float& dy = 1.0f,
                        const float& dz = 1.0f );

    void render_outline(void);

    void render(void);

    unsigned int size_x() const
    {
        return sx * scale;
    }
    unsigned int size_y() const
    {
        return sy * scale;
    }
    unsigned int size_z() const
    {
        return sz * scale;
    }
};






}

#endif // GL_VOLUMN_H

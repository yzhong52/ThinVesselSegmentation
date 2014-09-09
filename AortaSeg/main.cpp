#include <iostream>
#include <opencv2/core/core.hpp>

#include "../core/Data3D.h"
#include "../core/GLViewerCore.h"
#include "../core/ImageProcessing.h"

#include "../Vesselness/VesselnessTypes.h"

using namespace std;
using namespace cv;

/*This project will generate a mask of the Aorta*/

static const string dataname = "../temp/vessel3d_rd_sp";
static const string maskname = "../temp/vessel3d_rd_sp.aorta.mask.data";

bool generating_segmentation()
{
    Data3D<short> im_short;
    bool flag = im_short.load( dataname + ".data" );
    if( !flag ) return 0;

    cout << endl << endl << "Blur the data with Gaussian Filter... ";
    cout.flush();
    Data3D<short> im_short_blur;
    IP::GaussianBlur3D( im_short, im_short_blur, 61 );

    cout << endl << endl << "Generating a mask with thresholding... ";
    cout.flush();
    Data3D<unsigned char> mask;
    IP::threshold( im_short_blur, mask, short(6900) );

    cout << endl << endl << "Dialating the mask ...";
    cout.flush();
    IP::dilate( mask, 8 );

    cout << endl << endl << "A third of the volume are marked as aorta...";
    cout.flush();
    for( int z=0; z<mask.SZ()/4; z++ )
    {
        for( int y=0; y<mask.SY(); y++ )
        {
            for( int x=0; x<mask.SX(); x++ )
            {
                mask.at(x,y,z) = 255;
            }
        }
    }

    mask.save( maskname );

    return true;
}

bool validating_segmentation()
{
    bool flag = false;

    Data3D<unsigned char> im_mask;
    flag = im_mask.load( maskname );
    if( !flag ) return 0;

    cout << endl << "A third of the volume are marked as aorta..." << endl;
    for( int z=0; z<im_mask.SZ()/4; z++ )
    {
        for( int y=0; y<im_mask.SY(); y++ )
        {
            for( int x=0; x<im_mask.SX(); x++ )
            {
                im_mask.at(x,y,z) = 255;
            }
        }
    }

    im_mask.save( maskname );

    Data3D<short> im_short;
    flag = im_short.load( dataname + ".data" );
    if( !flag ) return 0;

    Data3D<Vesselness_Sig> vn_sig;
    vn_sig.load( dataname + ".et.vn_sig" );
    if( !flag ) return 0;
    Data3D<float> vn_float, vn_float2;
    vn_sig.copyDimTo(vn_float, 0 );
    IP::masking( vn_float, im_mask, vn_float2 );

    cout << endl << endl << "Generating image with mask... ";
    cout.flush();
    Data3D<short> im_short2;
    IP::masking( im_short, im_mask, im_short2 );

    GLViewerCore vis;

    vis.addObject( im_short,  GLViewer::Volumn::MIP );
    vis.addObject( vn_float2,  GLViewer::Volumn::MIP );
    vis.addObject( im_short2,  GLViewer::Volumn::MIP );
    vis.addObject( im_mask,  GLViewer::Volumn::MIP );
    vis.display( 1280, 800, 4 );

    return true;
}

int main()
{
    // TODO: This is for force the linking of OpenCV
    Mat m = Mat(1,1,CV_32F);
    imshow( "Temp", m );


    /*
    GLViewerCore vis;
    Data3D<short> im_short;
    bool flag = im_short.load( dataname + ".data" );
    if( !flag ) return 0;
    // im_short.remove_margin( Vec3i(200,200,200), Vec3i(0,0,650) );
    vis.addObject( im_short,  GLViewer::Volumn::MIP );

    cout << "Bluring image... " << endl;
    Data3D<short> im_short_blur;
    IP::GaussianBlur3D( im_short, im_short_blur, 5 );

    cout << "Thresholding image... " << endl;
    Data3D<unsigned char> mask( im_short.get_size() );
    #pragma omp parallel for
    for( int z=0; z<im_short.SZ(); z++ )
    {
        for( int y=0; y<im_short.SY(); y++ )
        {
            for( int x=0; x<im_short.SX(); x++ )
            {
                if( im_short.at(x,y,z) > 1800 && im_short.at(x,y,z) < 4000 )
                {
                    mask.at(x,y,z) = 255;
                }
            }
        }
    }
    vis.addObject( mask,  GLViewer::Volumn::MIP );

    cout << "Erosing image... " << endl;
    IP::erose( mask, 1 );
    vis.addObject( mask,  GLViewer::Volumn::MIP );

    vis.display( 1280, 800, 4 );
    /**/

    generating_segmentation();
    validating_segmentation();

    return 0;
}

#include "VesselDetector.h"
#include "Image3D.h"
#include "Kernel3D.h"
#include "ImageProcessing.h"
#include "VesselnessTypes.h"
#include "../EigenDecomp/eigen_decomp.h"

using namespace std;

Vesselness hessien_thread_func( const Data3D<float>& src,
                                const int& x, const int& y, const int& z,
                                const float& alpha, const float& beta, const float& gamma )
{

    ////////////////////////////////////////////////////////////////////
    // The following are being computed in this function
    // 1) derivative of images;
    // 2) Hessian matrix;
    // 3) Eigenvalue decomposition;
    // 4) vesselness measure.

    // 1) derivative of the image
    float im_dx2 = -2.0f * src.at(x,y,z) + src.at(x-1,y,z) + src.at(x+1,y,z);
    float im_dy2 = -2.0f * src.at(x,y,z) + src.at(x,y-1,z) + src.at(x,y+1,z);
    float im_dz2 = -2.0f * src.at(x,y,z) + src.at(x,y,z-1) + src.at(x,y,z+1);
    // 1) derivative of the image (alternative approach, the one above is more accurate)
    //float im_dx2 = -0.5f * src.at(x,y,z) + 0.25f * src.at(x-2,y,z) + 0.25f * src.at(x+2,y,z);
    //float im_dy2 = -0.5f * src.at(x,y,z) + 0.25f * src.at(x,y-2,z) + 0.25f * src.at(x,y+2,z);
    //float im_dz2 = -0.5f * src.at(x,y,z) + 0.25f * src.at(x,y,z-2) + 0.25f * src.at(x,y,z+2);

    float im_dxdy = (
                        + src.at(x-1, y-1, z)
                        + src.at(x+1, y+1, z)
                        - src.at(x-1, y+1, z)
                        - src.at(x+1, y-1, z) ) * 0.25f;
    float im_dxdz = (
                        + src.at(x-1, y, z-1)
                        + src.at(x+1, y, z+1)
                        - src.at(x+1, y, z-1)
                        - src.at(x-1, y, z+1) ) * 0.25f;
    float im_dydz = (
                        + src.at(x, y-1, z-1)
                        + src.at(x, y+1, z+1)
                        - src.at(x, y+1, z-1)
                        - src.at(x, y-1, z+1) ) * 0.25f;

    // 3) Eigenvalue decomposition
    // Reference: http://en.wikipedia.org/wiki/Eigenvalue_algorithm#3.C3.973_matrices
    // Given a real symmetric 3x3 matrix Hessian, compute the eigenvalues
    const float Hessian[6] = {im_dx2, im_dxdy, im_dxdz, im_dy2, im_dydz, im_dz2};

    float eigenvalues[3];
    float eigenvectors[3][3];
    eigen_decomp( Hessian, eigenvalues, eigenvectors );

    // order eigenvalues so that |lambda1| < |lambda2| < |lambda3|
    int i=0, j=1, k=2;
    if( abs(eigenvalues[i]) > abs(eigenvalues[j]) ) std::swap( i, j );
    if( abs(eigenvalues[i]) > abs(eigenvalues[k]) ) std::swap( i, k );
    if( abs(eigenvalues[j]) > abs(eigenvalues[k]) ) std::swap( j, k );

    // 4) vesselness measure
    Vesselness vn;
    // vesselness value
    if( eigenvalues[j] > 0 || eigenvalues[k] > 0 )
    {
        vn.rsp = 0.0f;
    }
    else
    {
        float lmd1 = abs( eigenvalues[i] );
        float lmd2 = abs( eigenvalues[j] );
        float lmd3 = abs( eigenvalues[k] );

        float A = (lmd3>1e-5) ? lmd2/lmd3 : 0;
        float B = (lmd2*lmd3>1e-5) ? lmd1 / sqrt( lmd2*lmd3 ) : 0;
        float S = sqrt( lmd1*lmd1 + lmd2*lmd2 + lmd3*lmd3 );
        vn.rsp = ( 1.0f-exp(-A*A/alpha) )* exp( B*B/beta ) * ( 1-exp(-S*S/gamma) );
    }

    // orientation of vesselness is corresponding to the eigenvector of the
    // smallest eigenvalue
    for( int d=0; d<3; d++ )
    {
        vn.dir[d]        = eigenvectors[i][d];
    }

    return vn;
}


bool VesselDetector::hessien( const Data3D<short>& src, Data3D<Vesselness>& dst,
                              int ksize, float sigma,
                              float alpha, float beta, float gamma )
{
    if( ksize!=0 && sigma<1e-3 ) // sigma is not set
    {
        sigma = 0.15f * ksize + 0.35f;
    }
    else if ( ksize==0 && sigma>1e-3 ) // size is not set
    {
        ksize = int( 6*sigma+1 );
        // make sure size is an odd number
        if ( ksize%2==0 ) ksize++;
    }
    else
    {
        std::cerr << "At lease size or sigma has to be set." << std::endl;
        return false;
    }

    Image3D<float> im_blur;
    bool flag = ImageProcessing::GaussianBlur3D( src, im_blur, ksize, sigma );
    smart_return( flag, "Gaussian Blur Failed.", false );

    im_blur *= sigma; //Normalizing for different scale

    dst.reset( im_blur.get_size(), Vesselness(0.0f) );

    #pragma omp parallel
    {
        #pragma omp for
        for( int z = 1; z < src.get_size_z()-1; z++ )
        {
            for( int y = 1; y < src.get_size_y()-1; y++ )
            {
                for( int x = 1; x < src.get_size_x()-1; x++ )
                {
                    dst.at(x, y, z) = hessien_thread_func( im_blur, x, y, z, alpha, beta, gamma);
                }
            }
        }
    }

    return true;
}



int VesselDetector::compute_vesselness(
    const Data3D<short>& src,							// INPUT
    Data3D<Vesselness_Sig>& dst,						// OUTPUT
    float sigma_from, float sigma_to, float sigma_step, // INPUT
    float alpha, float beta, float gamma )				// INPUT
{
    std::cout << "Computing Vesselness, it will take a while... " << std::endl;
    std::cout << "Vesselness will be computed from sigma = " << sigma_from << " to sigma = " << sigma_to << std::endl;


    dst.reset( src.get_size() ); // reszie data, and it will also be clear to zero

    // Error for input parameters
    smart_return( sigma_from < sigma_to,
                        "sigma_from should be smaller than sigma_to ", 0 );
    smart_return( sigma_step > 0,
                        "sigma_step should be greater than 0 ", 0 );

    int x,y,z;
    float max_sigma = sigma_from;
    float min_sigma = sigma_to;

    Data3D<Vesselness> temp;
    for( float sigma = sigma_from; sigma < sigma_to; sigma += sigma_step )
    {
        cout << '\r' << "Vesselness for sigma = " << sigma << "    " << "\b\b\b\b";
        cout.flush();

        VesselDetector::hessien( src, temp, 0, sigma, alpha, beta, gamma );

        const int margin = 1;
        for( z=margin; z<src.get_size_z()-margin; z++ )
        {
            for( y=margin; y<src.get_size_y()-margin; y++ )
            {
                for( x=margin; x<src.get_size_x()-margin; x++ )
                {
                    // Update vesselness if the new response is greater
                    if( dst.at(x, y, z).rsp < temp.at(x, y, z).rsp )
                    {
                        dst.at(x, y, z).rsp = temp.at(x, y, z).rsp;
                        dst.at(x, y, z).dir = temp.at(x, y, z).dir;
                        dst.at(x, y, z).sigma = sigma;
                        max_sigma = std::max( sigma, max_sigma );
                        min_sigma = std::min( sigma, min_sigma );
                    }
                }
            }
        }
    }

    std::cout << std::endl << "The range for sigma is [";
    std::cout << min_sigma << ", " << max_sigma << "]" << std::endl;
    std::cout << "Done. " << std::endl << std::endl;

    return 0;
}

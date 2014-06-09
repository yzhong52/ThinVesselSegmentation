#include "RingsReduction.h"
#include "ImageProcessing.h"
#include <opencv2/opencv.hpp>
#include <iostream>
using namespace cv;
using namespace std;



int RingsReduction::max_ring_radius( const cv::Vec2i& center, const cv::Vec2i& im_size )
{
    // four relative corner with respect to the centre of rings
    const Vec2i offsets[4] =
    {
        Vec2i(0          - center[0], 0          - center[1]),
        Vec2i(im_size[0] - center[0], im_size[1] - center[1]),
        Vec2i(0          - center[0], im_size[1] - center[1]),
        Vec2i(im_size[0] - center[0], 0          - center[1])
    };
    // maximum possible radius of the rings
    int max_radius = 0;
    for( int i=0; i<4; i++ )
    {
        int cur = offsets[i][0]*offsets[i][0] + offsets[i][1]*offsets[i][1];
        max_radius = max( max_radius, cur );
    }
    max_radius = int( sqrt(1.0f*max_radius) );

    return max_radius;
}


double RingsReduction::avgI_on_rings( const cv::Mat_<short>& m,
                                      const cv::Vec2i& ring_center,
                                      const double& r,
                                      const double& dr )
{
    // sum of intensities
    double sumI = 0.0;
    // number of pixels
    double pixel_count = 0;

    // center of the ring
    const int& center_x = ring_center[0];
    const int& center_y = ring_center[1];

    // image size
    const int& im_size_x = m.rows;
    const int& im_size_y = m.cols;

    // the range of the ring is [r_min,r_max], any pixels falls between this
    // range are considered as part of the ring
    const double r_min = r - 1;
    const double r_max = r + 1;
    const double r_min2 = r_min * r_min;
    const double r_max2 = r_max * r_max;

    // (x,y) pixel position with respect the center of the ring
    for( int x = 1; x<=r_max; x++ )
    {
        const int x2 = x * x;

        const int y_min = int( std::ceil(  sqrt( std::max(0.0, r_min2 - x2) ) ));
        const int y_max = int( std::floor( sqrt( r_max2 - x2 ) ) );

        for( int y=std::max(y_min, 1); y<=y_max; y++ )
        {
            const int y2 = y * y;

            // distance from the current point to the centre of the ring
            const double pixel_radius = sqrt( 1.0 * x2 + y2 );

            const double percentage = 1.0 - abs( pixel_radius - r);
            smart_assert( percentage>=0.0 && percentage <= 1.0,
                          "percentage should within range [0.0, 1.0]" );

            // 4 Quandrants
            for( int quadrant = 0; quadrant < 4; quadrant++ )
            {
                const int pixel_x = center_x + x * ((quadrant&&1)*2-1);
                const int pixel_y = center_y + y * ((quadrant>>1)*2-1);

                if( pixel_x >= 0 && pixel_x < im_size_x
                        && pixel_y >= 0 && pixel_y < im_size_y )
                {
                    sumI += m(pixel_y, pixel_x) * percentage;
                    pixel_count += percentage;
                }
            }
        }
    }

    // along 4 axis
    for( int i=0; i<4; i++ )
    {
        static const int aoffset[4][2] =
        {
            {-1, 0}, { 0,-1}, { 1, 0}, { 0, 1}
        };
        const int pixel_x = center_x + r * aoffset[i][0];
        const int pixel_y = center_y + r * aoffset[i][1];

        if( pixel_x >= 0 && pixel_x < im_size_x
                && pixel_y >= 0 && pixel_y < im_size_y )
        {
            sumI += m(pixel_y, pixel_x);
            pixel_count += 1.0;
        }
    }

    smart_assert( pixel_count > 1e-2,
                  "Pixel Count is too small. Potential bug. " << endl <<
                  "\t pixel_count: " << pixel_count << endl <<
                  "\t radius: " << r );


    return sumI / pixel_count;
}


void RingsReduction::correct_image( const Data3D<short>& src,
                                    Data3D<short>& dst,
                                    const vector<double>& correction,
                                    const int& slice,
                                    const cv::Vec2i& ring_center )
{
    dst.reset( src.get_size(), short(0) );

    for( int x=0; x<src.SX(); x++ )
    {
        for( int y=0; y<src.SY(); y++ )
        {
            for( int z=0; z<src.SZ(); z++ )
            {
                double diff_x = x - ring_center[0];
                double diff_y = y - ring_center[1];
                double radius = sqrt( diff_x*diff_x + diff_y*diff_y );
                int flo = (int) std::floor( radius );
                int cei = (int) std::ceil( radius );
                double c = 0;
                if( flo!=cei )
                {
                    c = correction[flo] * ( cei - radius ) +
                        correction[cei] * ( radius - flo );
                }
                else
                {
                    c = correction[flo];
                }
                dst.at(x,y,z) = src.at(x,y,z) - c;
            }
        }
    }
}

void RingsReduction::unname_method( const Data3D<short>& src, Data3D<short>& dst )
{
    smart_assert( &src!=&dst, "The destination file is the same as the orignal. " );

    // TODO: set the center as parameters
    const int center_x = 234; // 234;
    const int center_y = 270; // 270;

    // TODO
    const int center_z = src.SZ() / 2;

    const Mat_<short> slice = src.getMat(center_z);

    const Vec2i ring_center( center_x, center_y );
    const Vec2i im_size( slice.rows, slice.cols );

    int max_radius = max_ring_radius( ring_center, im_size );

    vector<double> correction( max_radius, 10 );

    // average intensity at radius r
    double avgIr = avgI_on_rings( slice, ring_center, 100 );

    for( int r = 1; r<=max_radius; r++ )
    {
        // average intensity at radius r + 1
        double avgIr1 = avgI_on_rings( slice, ring_center, r );
        correction[r] = avgIr1 - avgIr;
    }

    correct_image( src, dst, correction, center_z, ring_center );
}

// helper structure
typedef struct
{
    short diff;
    int var;
} Diff_Var;

// helper fucntion
short get_reduction( Diff_Var* diff_var, int count )
{
    ////////////////////////////
    //// Uncomment the following code if you want to use variance
    //// sort the data based on var
    //for( int i=0; i<count; i++ ){
    //	for( int j=i+1; j<count; j++ ){
    //		if( diff_var[i].var > diff_var[j].var )
    //			std::swap( diff_var[i], diff_var[j] );
    //	}
    //}
    //// discard 40% of the data (with big variances)
    // count = std::min( count, std::max( 20, count*4/5 ));

    // sort the data based on diff
    for( int i=0; i<count; i++ )
    {
        for( int j=i+1; j<count; j++ )
        {
            if( diff_var[i].diff > diff_var[j].diff )
                std::swap( diff_var[i], diff_var[j] );
        }
    }
    return diff_var[count/2].diff;
}

void RingsReduction::mm_filter( const Data3D<short>& im_src, Data3D<short>& dst )
{

    // TODO: set the center as parameters
    const int center_x = 234;
    const int center_y = 270;
    const int wsize = 15;


    if( &dst!=&im_src)
    {
        dst.resize( im_src.get_size() );
    }

    Data3D<short> mean( im_src.get_size() );
    IP::meanBlur3D( im_src, mean, wsize );
    //mean.save( "temp.mean.data" );
    //mean.load( "temp.mean.data" );
    //mean.show( "rd mean" );

    Data3D<int> diff( im_src.get_size() );
    subtract3D(im_src, mean, diff);
    //diff.save( "temp.diff.data", "rings reduction intermedia result - \
    //									Mean Blur with sigma = 19. Differnce between \
    //									original data. " );
    //diff.load( "temp.diff.data" );
    //diff.show( "rd diff" );

    //// Uncomment the following code if you want to use variance
    //Image3D<int> variance_sum( im.get_size() );
    //multiply3D(diff, diff, variance_sum);
    //Image3D<int> variance( im.get_size() );
    //IP::meanBlur3D( variance_sum, variance, wsize );

    // four relative corner with respect to the centre of rings
    Vec2i offsets[4] =
    {
        Vec2i(0                 -center_x, 0                 -center_y),
        Vec2i(im_src.get_size(0)-center_x, im_src.get_size(1)-center_y),
        Vec2i(0                 -center_x, im_src.get_size(1)-center_y),
        Vec2i(im_src.get_size(0)-center_x, 0                 -center_y)
    };
    // maximum possible radius of the rings
    int max_radius = 0;
    for( int i=0; i<4; i++ )
    {
        max_radius = max( max_radius, offsets[i][0]*offsets[i][0] + offsets[i][1]*offsets[i][1]);
    }
    max_radius = int( sqrt(1.0f*max_radius) );

    Diff_Var* diff_var = new Diff_Var[ int(4*M_PI*max_radius) ];

    // the ring reduction map: indicate whether a ring is stronger or weaker
    short* rdmap = new short[ max_radius ];
    memset( rdmap, 0, sizeof(short)*max_radius );

    const Vec3i& src_size = im_src.get_size();
    int x, y, z, r;
    for( z=0; z<src_size[2]; z++ )
    {
        // rings reduction is done slice by slice
        cout << '\r' << "Rings Reduction: " << 100 * z / src_size[2] << "%";
        cout.flush();

        rdmap[0] = diff.at(center_x, center_y, z);
        for( r=1; r<max_radius; r++ )
        {
            int count = 0;
            int r_min = r-1;
            int r_max = r+1;
            int r_min_square = r_min * r_min;
            int r_max_square = r_max * r_max;
            for( int x=1; x<=r_max; x++ )
            {
                int x_square = x * x;
                int y_min = int( ceil(  sqrt( max(0.0, 1.0*r_min_square - x_square) ) ));
                int y_max = int( floor( sqrt( 1.0*r_max_square - x_square ) ) );
                for( int y=max(y_min, 1); y<=y_max; y++ )
                {
                    int offset_x, offset_y;
                    // Quandrant 1
                    offset_x = center_x+x;
                    offset_y = center_y+y;
                    if( offset_x>=0 && offset_x<src_size[0] && offset_y>=0 && offset_y<src_size[1] )
                    {
                        diff_var[count].diff = diff.at(offset_x, offset_y, z);
                        //// Uncomment the following code if you want to use variance
                        //diff_var[count].var = variance.at(offset_x, offset_y, z);
                        count++;
                    }
                    // Quandrant 2
                    offset_x = center_x-x;
                    offset_y = center_y+y;
                    if( offset_x>=0 && offset_x<src_size[0] && offset_y>=0 && offset_y<src_size[1] )
                    {
                        diff_var[count].diff = diff.at(offset_x, offset_y, z);
                        //// Uncomment the following code if you want to use variance
                        //diff_var[count].var = variance.at(offset_x, offset_y, z);
                        count++;
                    }
                    // Quandrant 3
                    offset_x = center_x+x;
                    offset_y = center_y-y;
                    if( offset_x>=0 && offset_x<src_size[0] && offset_y>=0 && offset_y<src_size[1] )
                    {
                        diff_var[count].diff = diff.at(offset_x, offset_y, z);
                        //// Uncomment the following code if you want to use variance
                        //diff_var[count].var = variance.at(offset_x, offset_y, z);
                        count++;
                    }
                    // Quandrant 4
                    offset_x = center_x-x;
                    offset_y = center_y-y;
                    if( offset_x>=0 && offset_x<src_size[0] && offset_y>=0 && offset_y<src_size[1] )
                    {
                        diff_var[count].diff = diff.at(offset_x, offset_y, z);
                        //// Uncomment the following code if you want to use variance
                        //diff_var[count].var = variance.at(offset_x, offset_y, z);
                        count++;
                    }
                }
            }
            for( int i=max(r_min, 1) ; i<=r_max; i++ )
            {
                int offset_x, offset_y;
                // y > 0
                offset_x = center_x;
                offset_y = center_y+i;
                if( offset_y>=0 && offset_y<src_size[1] )
                {
                    diff_var[count].diff = diff.at(offset_x, offset_y, z);
                    //// Uncomment the following code if you want to use variance
                    //diff_var[count].var = variance.at(offset_x, offset_y, z);
                    count++;
                }
                // x > 0
                offset_x = center_x+i;
                offset_y = center_y;
                if( offset_x>=0 && offset_x<src_size[0] )
                {
                    diff_var[count].diff = diff.at(offset_x, offset_y, z);
                    //// Uncomment the following code if you want to use variance
                    //diff_var[count].var = variance.at(offset_x, offset_y, z);
                    count++;
                }
                // y < 0
                offset_x = center_x;
                offset_y = center_y-i;
                if( offset_y>=0 && offset_y<src_size[1] )
                {
                    diff_var[count].diff = diff.at(offset_x, offset_y, z);
                    //// Uncomment the following code if you want to use variance
                    //diff_var[count].var = variance.at(offset_x, offset_y, z);
                    count++;
                }
                // x < 0
                offset_x = center_x-i;
                offset_y = center_y;
                if( offset_x>=0 && offset_x<src_size[0] )
                {
                    diff_var[count].diff = diff.at(offset_x, offset_y, z);
                    //// Uncomment the following code if you want to use variance
                    //diff_var[count].var = variance.at(offset_x, offset_y, z);
                    count++;
                }
            }

            rdmap[r] = get_reduction( diff_var, count );
        } // end of loop r

        // remove ring for slice z
        for( y=0; y<src_size[1]; y++ )
        {
            for( x=0; x<src_size[0]; x++ )
            {
                // relative possition to the center of the ring
                int relative_x = x - center_x;
                int relative_y = y - center_y;
                if( relative_x==0 && relative_y==0 )
                {
                    // center of the ring
                    dst.at(x,y,z) = mean.at(x,y,z);
                    continue;
                }
                // radius of the ring
                float r = sqrt( float(relative_x*relative_x+relative_y*relative_y) );
                int floor_r = (int) floor(r);
                int ceil_r = (int) ceil(r);
                if( floor_r==ceil_r )
                {
                    dst.at(x,y,z) = im_src.at(x,y,z) - rdmap[ceil_r];
                }
                else
                {
                    dst.at(x,y,z) = im_src.at(x,y,z) - short( rdmap[ceil_r] * (r-floor_r) );
                    dst.at(x,y,z) = im_src.at(x,y,z) - short( rdmap[floor_r] * (ceil_r-r) );
                }
            }
        }
    }
    cout << endl;

    delete[] rdmap;
    delete[] diff_var;
}

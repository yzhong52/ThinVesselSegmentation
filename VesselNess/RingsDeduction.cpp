#include "RingsDeduction.h"
#include "stdafx.h"
#include "Image3D.h"
#include "ImageProcessing.h"
#define _USE_MATH_DEFINES
#include <math.h>


typedef struct
{
    short diff;
    int var;
} Diff_Var;

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

void RingsDeduction::mm_filter( Data3D<short>& im_src, const int& wsize,
                                const int& center_x , const int& center_y )
{
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
                    im_src.at(x,y,z) = mean.at(x,y,z);
                    continue;
                }
                // radius of the ring
                float r = sqrt( float(relative_x*relative_x+relative_y*relative_y) );
                int floor_r = (int) floor(r);
                int ceil_r = (int) ceil(r);
                if( floor_r==ceil_r )
                {
                    im_src.at(x,y,z) -= rdmap[ceil_r];
                }
                else
                {
                    im_src.at(x,y,z) -= short( rdmap[ceil_r] * (r-floor_r) );
                    im_src.at(x,y,z) -= short( rdmap[floor_r] * (ceil_r-r) );
                }
            }
        }
    }
    cout << endl;

    delete[] rdmap;
    delete[] diff_var;
}


void RingsDeduction::gm_filter( Data3D<short>& im_src, const int& wsize,
                                const int& center_x , const int& center_y )
{

    Data3D<short> mean( im_src.get_size() );
    //Data3D<float> mean_float( im_src.get_size() );
    //cout << "Blurring Image With Gaussian Filter." << endl;
    //IP::GaussianBlur3D( im_src, mean_float, wsize, (wsize-1)/6 );
    //mean_float.convertTo( mean );
    //mean.save( "rd_gblur.21.data", "rings reduction intermedia result - Gaussian Blur with sigma = 21. " );
    mean.load( "rd_gblur.21.data" );
    //mean.show( "rd_gblur.21.data", im_src.SZ()/2 );

    Data3D<int> diff( im_src.get_size() );
    //subtract3D(im_src, mean, diff);
    //diff.save( "rd_gblur.diff.21.data", "rings reduction intermedia result - \
    //									Gaussian Blur with sigma = 21. Differnce between \
    //									original data. " );
    diff.load( "rd_gblur.diff.21.data" );
    //diff.show( "rd_gblur.diff.21.data" );


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
                    im_src.at(x,y,z) = (short) mean.at(x,y,z);
                    continue;
                }
                // radius of the ring
                float r = sqrt( float(relative_x*relative_x+relative_y*relative_y) );
                int floor_r = (int) floor(r);
                int ceil_r = (int) ceil(r);
                if( floor_r==ceil_r )
                {
                    im_src.at(x,y,z) -= rdmap[ceil_r];
                }
                else
                {
                    im_src.at(x,y,z) -= short( rdmap[ceil_r] * (r-floor_r) );
                    im_src.at(x,y,z) -= short( rdmap[floor_r] * (ceil_r-r) );
                }
            }
        }
    }
    cout << endl;

    delete[] rdmap;
    delete[] diff_var;
}

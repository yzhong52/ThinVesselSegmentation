#include "RingsReduction.h"

#include <opencv2/opencv.hpp>
#include <iostream>

#include "ImageProcessing.h"
#include "CVPlot.h"

using namespace cv;
using namespace std;



float RingsReduction::max_ring_radius( const cv::Vec2f& center,
                                       const cv::Vec2f& im_size )
{
    // four relative corner with respect to the centre of rings
    const Vec2f corner[4] =
    {
        Vec2f(0          - center[0], 0          - center[1]),
        Vec2f(im_size[0] - center[0], im_size[1] - center[1]),
        Vec2f(0          - center[0], im_size[1] - center[1]),
        Vec2f(im_size[0] - center[0], 0          - center[1])
    };

    // maximum possible radius of the rings
    float max_radius_squre = 0;
    for( int i=0; i<4; i++ )
    {
        float current = corner[i][0]*corner[i][0] + corner[i][1]*corner[i][1];
        max_radius_squre = max( max_radius_squre, current );
    }
    return sqrt( max_radius_squre );
}


double RingsReduction::avgI_on_rings( const cv::Mat_<short>& m,
                                      const cv::Vec2f& ring_center,
                                      const int& rid,
                                      const double& dr )
{
    smart_assert( dr>0, "dr indicates the thickness of the rings. \
                 It should be greater than 0. " );

    // sum of intensities
    double sumI = 0.0;

    // number of pixels
    double pixel_count = 0;

    // center of the ring
    const float& center_x = ring_center[0];
    const float& center_y = ring_center[1];

    // image size
    const int& im_size_x = m.rows;
    const int& im_size_y = m.cols;

    // the range of the ring is [r_min,r_max], any pixels falls between this
    // range are considered as part of the ring
    double radius = rid * dr;
    const double r_min = radius - dr;
    const double r_max = radius + dr;
    const double r_min2 = r_min * r_min;
    const double r_max2 = r_max * r_max;

    // (x,y) pixel position with respect the center of the ring
    for( double x = 1; x<=r_max; x++ )
    {
        const double x2 = x * x;

        const double y_min = sqrt( std::max(0.0, r_min2 - x2) );
        const double y_max = sqrt( r_max2 - x2 );

        for( double y=std::max(y_min, 1.0); y<=y_max; y++ )
        {
            const double y2 = y * y;

            // distance from the current point to the centre of the ring
            const double pixel_radius = sqrt( x2 + y2 );

            const double dist_2_ring = abs( pixel_radius - radius );

            if( dist_2_ring>dr ) continue;

            const double percentage = 1.0 - dist_2_ring/dr;

            // 4 Quandrants
            for( int quadrant = 0; quadrant < 4; quadrant++ )
            {

                const int pixel_x = int( center_x + x * ((quadrant&&1)*2-1) );
                const int pixel_y = int( center_y + y * ((quadrant>>1)*2-1) );

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
        const int pixel_x = int( center_x + radius * aoffset[i][0] );
        const int pixel_y = int( center_y + radius * aoffset[i][1] );

        if( pixel_x >= 0 && pixel_x < im_size_x
                && pixel_y >= 0 && pixel_y < im_size_y )
        {
            sumI += m(pixel_y, pixel_x);
            pixel_count += 1.0;
        }
    }

    if( pixel_count > 1e-2 ) return sumI / pixel_count;
    else return 0.0;
}


void RingsReduction::sijbers( const Data3D<short>& src, Data3D<short>& dst,
                              std::vector<double>* pCorrection )
{
    // TODO: set the center as parameters
    const float center_x = 234;
    const float center_y = 270;
    const int wsize = 15;

    if( &dst!=&src)
    {
        dst.resize( src.get_size() );
    }

    // blur the image with mean blur
    Data3D<short> mean( src.get_size() );
    IP::meanBlur3D( src, mean, wsize );

    Data3D<int> diff( src.get_size() );
    subtract3D( src, mean, diff );

    /// TODO: Uncomment the following code if you want to use variance
    //Data3D<int> variance_sum( im.get_size() );
    //multiply3D(diff, diff, variance_sum);
    //Data3D<int> variance( im.get_size() );
    //IP::meanBlur3D( variance_sum, variance, wsize );

    const Vec2f ring_center( center_x, center_y );
    const Vec2f im_size( (float) src.SX(), (float) src.SY() );

    const float max_radius = max_ring_radius( ring_center, im_size );
    const float dr = 1;
    const unsigned num_of_rings = unsigned( max_radius / dr );

    for( int z=0; z<src.SZ(); z++ )
    {
        z = src.SZ() / 2;
        vector<double> correction( num_of_rings, 0 );

        // rings reduction is done slice by slice
        cout << '\r' << "Rings Reduction: " << 100 * z / src.SZ() << "%";
        cout.flush();

        const Mat_<int> m = diff.getMat(z);
        for( unsigned ri = 0; ri<num_of_rings-1; ri++ )
        {
            correction[ri] = med_on_ring( m, ring_center, ri, dr );
        }

        correct_image( src, dst, correction, z, ring_center, dr );
        if( pCorrection ) *pCorrection = correction;
        break;
    }
    cout << endl;
}



void RingsReduction::correct_image( const Data3D<short>& src,
                                    Data3D<short>& dst,
                                    const vector<double>& correction,
                                    const int& slice,
                                    const Vec2i& ring_center,
                                    const double& dr )
{
    dst.reset( src.get_size(), short(0) );

    const int& z = slice;
    for( int x=0; x<src.SX(); x++ )
    {
        for( int y=0; y<src.SY(); y++ )
        {

            const double diff_x = x - ring_center[0];
            const double diff_y = y - ring_center[1];
            const double radius = sqrt( diff_x*diff_x + diff_y*diff_y );

            // For any rid that bigger than the size of the correction vector,
            // pretend that it is the most outter ring (rid = correction.size()-1).
            // This assumption may
            // result in a few of the pixels near the corner of the image corner
            // being ignored. But it will prevend the function from crashing if
            // it was not used properly.
            const double rid = min( radius/dr, (double) correction.size()-1 );

            const int flo = (int) std::floor( rid );
            const int cei = (int) std::ceil( rid );
            double c = 0;
            if( flo!=cei )
            {
                c = correction[flo] * ( cei - rid ) +
                    correction[cei] * ( rid - flo );
            }
            else
            {
                c = correction[flo];
            }
            dst.at(x,y,z) = short( src.at(x,y,z) - c );
        }
    }
}

void RingsReduction::unname_method( const Data3D<short>& src, Data3D<short>& dst )
{
    smart_assert( 0, "deprecated func" );
    smart_assert( &src!=&dst,
                  "The destination file is the same as the orignal. " );

    // TODO: set the center as parameters
    const int center_x = 234; // 234;
    const int center_y = 270; // 270;

    // TODO: do it on a 3D volume
    const int center_z = src.SZ() / 2;

    const Mat_<short> slice = src.getMat(center_z);

    const Vec2f ring_center( center_x, center_y );
    const Vec2f im_size( (float) slice.rows, (float) slice.cols );

    float max_radius = max_ring_radius( ring_center, im_size );

    const float dr = 1;
    int num_of_rings = int( max_radius / dr );

    vector<double> correction( num_of_rings, 10 );

    // average intensity at radius r
    double avgIr = avgI_on_rings( slice, ring_center, 100, dr );

    for( int ri = 0; ri<=num_of_rings; ri++ )
    {
        // average intensity at radius r * dr
        double avgIr1 = avgI_on_rings( slice, ring_center, ri, dr );
        correction[ri] = avgIr1 - avgIr;
    }

    // correct_image( src, dst, correction, center_z, ring_center, dr );
}


void RingsReduction::polarRD( const Data3D<short>& src, Data3D<short>& dst,
                              const PolarRDOption& o, const float dr,
                              vector<double>* pCorrection )
{
    smart_assert( &src!=&dst,
                  "The destination file is the same as the orignal. " );

    // TODO: set the center as parameters
    // TODO: change them to float
    const float center_x = 234; // 234;
    const float center_y = 270; // 270;

    // TODO: do it on a 3D volume
    const int center_z = src.SZ() / 2;

    const Vec2f ring_center( center_x, center_y );
    const Vec2f im_size( (float)src.SX(), (float)src.SY() );

    const float max_radius = max_ring_radius( ring_center, im_size );

    const int num_of_rings = int( max_radius / dr );

    double (*diff_func)(const cv::Mat_<short>&,
                        const cv::Vec2f&,
                        const int&,
                        const int&,
                        const double&) = nullptr;
    switch (o )
    {
    case AVG_DIFF:
        diff_func = &avg_diff_v2;
        break;
    case MED_DIFF:
        diff_func = &med_diff_v2;
        break;
    default:
        cerr << "Undefined method option. " << endl;
        break;
    }

    // compute correction vector
    vector<double> correction( num_of_rings, 0 );
    for( int ri = 0; ri<num_of_rings-1; ri++ )
    {
        correction[ri] = diff_func( src.getMat(center_z),
                                    ring_center, ri, 100, dr );
    }

    correct_image( src, dst, correction, center_z, ring_center, dr );

    if( pCorrection!=nullptr ) *pCorrection = correction;
}



void RingsReduction::polarRD_accumulate( const Data3D<short>& src, Data3D<short>& dst,
                         const PolarRDOption& o, const float dr,
                         std::vector<double>* pCorrection )
{
    smart_assert( &src!=&dst,
                  "The destination file is the same as the orignal. " );

    // TODO: set the center as parameters
    // TODO: change them to float
    const float center_x = 234; // 234;
    const float center_y = 270; // 270;

    // TODO: do it on a 3D volume
    const unsigned center_z = src.SZ() / 2;

    const Vec2f ring_center( center_x, center_y );
    const Vec2f im_size( (float)src.SX(), (float)src.SY() );

    const float max_radius = max_ring_radius( ring_center, im_size );

    const unsigned num_of_rings = unsigned( max_radius / dr );

    double (*diff_func)(const cv::Mat_<short>&,
                        const cv::Vec2f&,
                        const int&,
                        const int&,
                        const double&) = nullptr;

    switch (o )
    {
    case MED_DIFF:
        diff_func = &med_diff;
        break;
    default:
        diff_func = &med_diff;
        cerr << "Undefined method option. " << endl;
        break;
    }

    // compute correction vector
    vector<double> correction( num_of_rings, 0 );
    for( unsigned ri = 0; ri<num_of_rings-1; ri++ )
    {
        correction[ri] = diff_func( src.getMat(center_z),
                                    ring_center, ri, ri+1, dr );
    }

    for( unsigned ri = num_of_rings-2; ri>0; ri-- ){
        correction[ri] += correction[ri+1];
    }

    correct_image( src, dst, correction, center_z, ring_center, dr );

    if( pCorrection!=nullptr ) *pCorrection = correction;

}

double RingsReduction::interpolate( const cv::Mat_<short>& m, double x, double y )
{
    const int fx = (int) floor( x );
    const int cx = (int) ceil( x );
    const int fy = (int) floor( y );
    const int cy = (int) ceil( y );

    smart_assert( fx>=0 && cx<m.cols && fy>=0 && cy<m.rows,
                  "Invalid input image position. Please call the following " <<
                  "fucntion before computing interpolation. " << endl <<
                  "\t bool isvalid( cv::Mat_<short>&, double, double ); " );

    if( fx==cx && fy==cy )
    {
        return m(fy, fx);
    }
    else if( fx==cx )
    {
        return m(fy, fx) * (cy - y) +
               m(cy, fx) * (y - fy);
    }
    else if ( fy==cy )
    {
        return m(fy, fx) * (cx - x) +
               m(fy, cx) * (x - fx);
    }
    else
    {
        return m(fy, fx) * (cx - x) * (cy - y) +
               m(cy, fx) * (cx - x) * (y - fy) +
               m(fy, cx) * (x - fx) * (cy - y) +
               m(cy, cx) * (x - fx) * (y - fy);
    }
}

double RingsReduction::avg_diff( const cv::Mat_<short>& m,
                                 const cv::Vec2f& ring_center,
                                 const int& rid1,
                                 const int& rid2,
                                 const double& dr )
{
    // radius of the two circles
    const double radius  = rid1 * dr;
    const double radius1 = rid2 * dr;

    // the number of pixels on the circumference approximatly
    const double bigger = std::max( radius, radius1 );
    const int circumference = max( 8, int( 2 * M_PI * bigger ) );

    double sum = 0.0;
    int count = 0;
    for( int i=0; i<circumference; i++ )
    {
        // angle in radian
        const double angle = 2 * M_PI * i / circumference;
        const double sin_angle = sin( angle );
        const double cos_angle = cos( angle );

        // image possition for inner circle
        const double x = radius * cos_angle + ring_center[0];
        const double y = radius * sin_angle + ring_center[1];

        // image position for outter circle
        const double x1 = radius1 * cos_angle + ring_center[0];
        const double y1 = radius1 * sin_angle + ring_center[1];

        if( isvalid( m, x1, y1) && isvalid( m, x, y) )
        {
            const double val  = interpolate( m, x, y );
            const double val1 = interpolate( m, x1, y1 );
            sum += val - val1;
            count++;
        }
    }

    return (count>0) ? sum/count : 0;
}

double RingsReduction::med_diff( const cv::Mat_<short>& m,
                                 const cv::Vec2f& ring_center,
                                 const int& rid1,
                                 const int& rid2,
                                 const double& dr )
{
    // radius of the two circles
    const double radius  = rid1 * dr;
    const double radius1 = rid2 * dr;

    // the number of pixels on the circumference approximatly
    const double bigger = std::max( radius, radius1 );
    const int circumference = max( 8, int( 2 * M_PI * bigger ) );

    std::vector<double> diffs(1, 0);
    for( int i=0; i<circumference; i++ )
    {
        // angle in radian
        const double angle = 2 * M_PI * i / circumference;
        const double sin_angle = sin( angle );
        const double cos_angle = cos( angle );

        // image possition for inner circle
        const double x = radius * cos_angle + ring_center[0];
        const double y = radius * sin_angle + ring_center[1];

        // image position for outter circle
        const double x1 = radius1 * cos_angle + ring_center[0];
        const double y1 = radius1 * sin_angle + ring_center[1];

        if( isvalid( m, x1, y1) && isvalid( m, x, y) )
        {
            const double val  = interpolate( m, x, y );
            const double val1 = interpolate( m, x1, y1 );
            diffs.push_back( val - val1 );
        }
    }

    return median( diffs );
}

double RingsReduction::avg_diff_v2( const cv::Mat_<short>& m,
                                    const cv::Vec2f& ring_center,
                                    const int& rid1,
                                    const int& rid2,
                                    const double& dr )
{
    const double avg1 = avg_on_ring( m, ring_center, rid1, dr );
    const double avg2 = avg_on_ring( m, ring_center, rid2, dr );
    return avg1 - avg2;
}


double RingsReduction::med_diff_v2( const cv::Mat_<short>& m,
                                    const cv::Vec2f& ring_center,
                                    const int& rid1,
                                    const int& rid2,
                                    const double& dr )
{
    const double med1 = med_on_ring( m, ring_center, rid1, dr );
    const double med2 = med_on_ring( m, ring_center, rid2, dr );
    return med1 - med2;
}



double RingsReduction::avg_on_ring( const cv::Mat_<short>& m,
                                    const cv::Vec2f& ring_center,
                                    const int& rid,
                                    const double& dr)
{
    // radius of the circle
    const double radius = rid * dr;

    // the number of pixels on the circumference approximatly
    const int circumference = max( 8, int( 2 * M_PI * radius ) );

    int count = 0;
    double sum = 0.0;

    for( int i=0; i<circumference; i++ )
    {
        // angle in radian
        const double angle = 2 * M_PI * i / circumference;
        const double sin_angle = sin( angle );
        const double cos_angle = cos( angle );

        // image possition for inner circle
        const double x = radius * cos_angle + ring_center[0];
        const double y = radius * sin_angle + ring_center[1];

        if( isvalid( m, x, y) )
        {
            sum += interpolate( m, x, y );
            count++;
        }
    }

    return (count>0) ? sum/count : 0;
}



double RingsReduction::median( std::vector<double>& values )
{
    std::sort( values.begin(), values.end() );

    const double numVal = 0.5 * (double) values.size();
    const int id1 = (int) std::floor( numVal );
    const int id2 = (int) std::ceil(  numVal );
    if( id1 == id2 )
    {
        return values[id1];
    }
    else
    {
        return 0.5 * ( values[id1] + values[id2] );
    }
}

std::vector<double> RingsReduction::distri_of_diff( const cv::Mat_<short>& m,
                                const cv::Vec2f& ring_center,
                                const int& rid1, const int& rid2,
                                const double& dr )
{
    // radius of the two circles
    const double radius  = rid1 * dr;
    const double radius1 = rid2 * dr;

    // the number of pixels on the circumference approximatly
    const double bigger = std::max( radius, radius1 );
    const int circumference = max( 8, int( 2 * M_PI * bigger ) );

    std::vector<double> diffs(1, 0);
    for( int i=0; i<circumference; i++ )
    {
        // angle in radian
        const double angle = 2 * M_PI * i / circumference;
        const double sin_angle = sin( angle );
        const double cos_angle = cos( angle );

        // image possition for inner circle
        const double x = radius * cos_angle + ring_center[0];
        const double y = radius * sin_angle + ring_center[1];

        // image position for outter circle
        const double x1 = radius1 * cos_angle + ring_center[0];
        const double y1 = radius1 * sin_angle + ring_center[1];

        if( isvalid( m, x1, y1) && isvalid( m, x, y) )
        {
            const double val  = interpolate( m, x, y );
            const double val1 = interpolate( m, x1, y1 );
            diffs.push_back( val - val1 );
        }
    }
    return diffs;
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
    const int center_x = 234.0;
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
    //diff.save( "temp.diff.data", "rings reduction intermedia result -
    //									Mean Blur with sigma = 19. Differnce between
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
    max_radius = int( sqrt(1.0*max_radius) );

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

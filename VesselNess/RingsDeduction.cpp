#include "RingsDeduction.h"
#include "stdafx.h"
#include "Image3D.h"
#include "ImageProcessing.h"
#define _USE_MATH_DEFINES
#include <math.h>


typedef struct{
	short diff;
	int var;
} Diff_Var;



short get_reduction( Diff_Var* diff_var, int count ){
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
	for( int i=0; i<count; i++ ){
		for( int j=i+1; j<count; j++ ){
			if( diff_var[i].diff > diff_var[j].diff )
				std::swap( diff_var[i], diff_var[j] );
		}
	}
	return diff_var[count/2].diff;
}

void RingsDeduction::mm_filter( Image3D<short>& im, const int& wsize ){
	// center of the rings
	static const int center_y = 270;
	static const int center_x = 234;

	Image3D<short> mean( im.get_size() );
	IP::meanBlur3D( im, mean, wsize );
	// IP::GaussianBlur3D( src, mean, 15); // TODO: I amy try Gaussian later
	//mean.showData("Blured Image", 25);

	Image3D<int> diff( im.get_size() );
	subtract3D(im, mean, diff);
	//diff.showData("SRC-Mean Diff", 25);
	mean.reset(Vec3i(0,0,0));

	//Image3D<int> variance_sum( im.get_size() );
	//multiply3D(diff, diff, variance_sum);
	////variance_sum.showData("Variance", 25);

	//Image3D<int> variance( im.get_size() );
	//IP::meanBlur3D( variance_sum, variance, wsize );
	////variance.showData("Variance", 25);

	// maximum possible radius of the data
	int max_radius =  center_x*center_x+center_y*center_y;
	Vec2i offsets[3] = {
		Vec2i(im.get_size(0)-center_x, im.get_size(1)-center_y),
		Vec2i(center_x,                im.get_size(1)-center_y),
		Vec2i(im.get_size(0)-center_x, center_y)
	};
	for( int i=0; i<3; i++ ){
		max_radius = max( max_radius, 
			offsets[i][0]*offsets[i][0] + offsets[i][1]*offsets[i][1]);
	}
	max_radius = int( sqrt(1.0f*max_radius) );

	Diff_Var* diff_var = new Diff_Var[ int(4*M_PI*max_radius) ];

	// the ring reduction map. all so important
	short* rdmap = new short[max_radius];
	memset( rdmap, 0, sizeof(short)*max_radius );

	const Vec3i& src_size = im.get_size();
	int x, y, z, r;
	for( z=0; z<src_size[2]; z++ ) {
		cout << '\r' << "Rings Reduction: " << 100 * z / src_size[2] << "%";
		rdmap[0] = diff.at(center_x, center_y, z);
		for( r=1; r<max_radius; r++ ){
			int count = 0;
			int r_min = r-1;
			int r_max = r+1;
			int r_min_square = r_min * r_min;
			int r_max_square = r_max * r_max;
			for( int x=1; x<=r_max; x++ ) {
				int x_square = x * x;
				int y_min = int( ceil(  sqrt( max(0.0, 1.0*r_min_square - x_square) ) ));
				int y_max = int( floor( sqrt( 1.0*r_max_square - x_square ) ) );
				for( int y=max(y_min, 1); y<=y_max; y++ ){
					int offset_x, offset_y;
					// Quandrant 1
					offset_x = center_x+x;
					offset_y = center_y+y;
					if( offset_x>=0 && offset_x<src_size[0] && offset_y>=0 && offset_y<src_size[1] ){
						diff_var[count].diff = diff.at(offset_x, offset_y, z);
						//diff_var[count].var = variance.at(offset_x, offset_y, z);
						count++;
					}
					// Quandrant 2
					offset_x = center_x-x;
					offset_y = center_y+y;
					if( offset_x>=0 && offset_x<src_size[0] && offset_y>=0 && offset_y<src_size[1] ){
						diff_var[count].diff = diff.at(offset_x, offset_y, z);
						//diff_var[count].var = variance.at(offset_x, offset_y, z);
						count++;
					}
					// Quandrant 3
					offset_x = center_x+x;
					offset_y = center_y-y;
					if( offset_x>=0 && offset_x<src_size[0] && offset_y>=0 && offset_y<src_size[1] ){
						diff_var[count].diff = diff.at(offset_x, offset_y, z);
						//diff_var[count].var = variance.at(offset_x, offset_y, z);
						count++;
					}
					// Quandrant 4
					offset_x = center_x-x;
					offset_y = center_y-y;
					if( offset_x>=0 && offset_x<src_size[0] && offset_y>=0 && offset_y<src_size[1] ){
						diff_var[count].diff = diff.at(offset_x, offset_y, z);
						//diff_var[count].var = variance.at(offset_x, offset_y, z);
						count++;
					}
				}
			}
			for( int i=max(r_min, 1) ; i<=r_max; i++ ){
				int offset_x, offset_y;
				// y > 0
				offset_x = center_x;
				offset_y = center_y+i;
				if( offset_y>=0 && offset_y<src_size[1] ){
					diff_var[count].diff = diff.at(offset_x, offset_y, z);
					//diff_var[count].var = variance.at(offset_x, offset_y, z);
					count++;
				}
				// x > 0
				offset_x = center_x+i;
				offset_y = center_y;
				if( offset_x>=0 && offset_x<src_size[0] ){
					diff_var[count].diff = diff.at(offset_x, offset_y, z);
					//diff_var[count].var = variance.at(offset_x, offset_y, z);
					count++;
				}
				// y < 0
				offset_x = center_x;
				offset_y = center_y-i;
				if( offset_y>=0 && offset_y<src_size[1] ){
					diff_var[count].diff = diff.at(offset_x, offset_y, z);
					//diff_var[count].var = variance.at(offset_x, offset_y, z);
					count++;
				}
				// x < 0
				offset_x = center_x-i;
				offset_y = center_y;
				if( offset_x>=0 && offset_x<src_size[0] ){
					diff_var[count].diff = diff.at(offset_x, offset_y, z);
					//diff_var[count].var = variance.at(offset_x, offset_y, z);
					count++;
				}
			}

			rdmap[r] = get_reduction( diff_var, count );
		} // end of loop r

		// remove ring for slice z
		for( y=0; y<src_size[1]; y++ ) {
			for( x=0; x<src_size[0]; x++ ) {
				// relative possition to the center of the ring
				int relative_x = x - center_x;
				int relative_y = y - center_y;
				float r = sqrt( float(relative_x*relative_x+relative_y*relative_y) );
				float floor_r = floor(r);
				float ceil_r = ceil(r);
				if( int(floor_r)==int(ceil_r) ) {
					im.at(x,y,z) -= rdmap[int(ceil_r)];
				} else {
					im.at(x,y,z) -= short( rdmap[int(ceil_r)] * (r-floor_r) );
					im.at(x,y,z) -= short( rdmap[int(floor_r)] * (ceil_r-r) );
				}
			}
		}
	}
	cout << endl;

	delete[] rdmap;
	delete[] diff_var;
}
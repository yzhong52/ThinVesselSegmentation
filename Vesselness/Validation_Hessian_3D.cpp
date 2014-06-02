#include "stdafx.h"
#include "Validation.h"
#include "Image3D.h"
#include "Kernel3D.h"
#include "ImageProcessing.h"
#include "VesselDetector.h"
#include "Viewer.h"
#include "VesselNess.h"


void Validation::construct_tube( Data3D<short>& image ) {
	image.reset( Vec3i(150, 200, 150), 500*25  );

	// center of the vessel
	const Vec3i center[5] = {
		Vec3i( 50, 0, 50),
		Vec3i(150, 0, 50),
		Vec3i(250, 0, 50),
		Vec3i(350, 0, 50),
		Vec3i(450, 0, 50)
	};
	// radius of vessels
	// const float radius[5] = { 0.5f, 1.5f, 2.5f, 3.5f, 4.5f };
	const float radius[5] = { 3.0f, 7.0f, 3.0f, 4.0f, 5.0f };

	int x, y, z;

	// Yuchen: There is a bug in the following code
	//for( int i=0; i<5; i++ ) {
	//	const float& r = radius[i];
	//	float r2 = r*r;
	//	int c_r = (int) ceil(r+1);
	//	for( y=0; y<image.get_size_y(); y++ )
	//	{
	//		for( z=0; z<=c_r; z++ )
	//		{
	//			int x_max = (int) ceil( sqrt( max(0.0f, 1.0f*r2-z*z)) );
	//			for( x=-x_max; x<=x_max; x++ )
	//			{
	//				// the distance from the current pixel to the center of the vessel
	//				float dis_center = sqrt( 1.0f*x*x + z*z );
	//				// compute and assign the value
	//				float ratio = r+0.5f-dis_center;
	//				if( ratio>1.0f ) {
	//					image.at(x+center[i][0], z+center[i][2], y) = MAX_SHORT;
	//				} else if (ratio<0.0f) {
	//					image.at(x+center[i][0], z+center[i][2], y) = 0;
	//				} else {
	//					image.at(x+center[i][0], z+center[i][2], y) = short( MAX_SHORT * ratio );
	//				}
	//				/*
	//				else if( ratio<0.0f ) ratio = 0.0f;
	//				image.at(x+center[i][0], z+center[i][2], y)
	//					= short( max( 0.0f, min(1.0f, r+0.5f-dis_center)) * MAX_SHORT );
	//				*/
	//			}
	//		}
	//	}
	//}

	for( int i=0; i<5; i++ ) {
		for( y=0; y<image.get_size_y(); y++ ) {
			for( z=-20; z<20; z++ ) {
				for( x=-20; x<20; x++ ){
					if( x+center[i][0] <= 0 ) continue;
					if( x+center[i][0] >=image.SX() ) continue;

					if( z+center[i][0] <= 0 ) continue;
					if( z+center[i][0] >=image.SZ() ) continue;

					float dis_center = sqrt( 1.0f*x*x + z*z );
					float ratio = radius[i] + 0.5f - dis_center;
					if( ratio>1.0f ) {
						image.at(x+center[i][0], y, z+center[i][2]) = MAX_SHORT;
					} else if( ratio>0.0f ) {
						image.at(x+center[i][0], y, z+center[i][2]) = short( MAX_SHORT * ratio ) + 500;
					}
				}
			}
		}
	}
}


void Validation::construct_tube2( Data3D<short>& image ) {
	image.reset( Vec3i(150, 150, 150) );

	// center of the vessel
	const Vec3i center[1] = {
		Vec3i( 75, 75, 75 )
	};
	// radius of vessels
	const float radius[1] = { 5.0f };

	int x, y, z;
	for( int i=0; i<1; i++ ) {
		for( y=0; y<image.get_size_y(); y++ ) {
			for( z=-20; z<20; z++ ) {
				for( x=-20; x<20; x++ ){
					float dis_center = sqrt( 1.0f*x*x + z*z );
					float ratio = radius[i] + 0.5f - dis_center;
					if( ratio>1.0f ) {
						image.at(x+center[i][0], y, z+center[i][2]) = MAX_SHORT;
					} else if( ratio>0.0f ) {
						image.at(x+center[i][0], y, z+center[i][2]) = short( MAX_SHORT * ratio );
					}
				}
			}
		}
	}
}

#include "stdafx.h"
#include "Validation.h"
#include "Image3D.h"
#include "Kernel3D.h"
#include "ImageProcessing.h"
#include "VesselDetector.h"
#include "Viewer.h"
#include "Vesselness.h"


void Validation::construct_tube( Data3D<short>& image ) {
	image.reset( Vec3i(500, 100, 100) );

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
	const float radius[5] = { 1.0f, 2.0f, 3.0f, 4.0f, 5.0f };

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

bool Validation::Hessian_3D(void){
	// generate a vessel here
	Image3D<short> image;
	construct_tube( image );
	// image.saveVideo( "3d tubes.avi" );

	const int SGNUM = 6;
	// float sigmas[4] = {3.0f, 5.0f, 7.0f, 9.0f};
	// float sigma[4] = {1.5f, 3.5f, 5.5f, 7.5f};
	// float sigma[4] = {1.0f, 3.0f, 5.0f, 7.0f};
	// float sigma[4] = { 0.5f, 1.5f, 2.5f, 3.5f };
	// float sigma[5] = { 0.5f, 1.0f, 1.5f, 2.0f, 2.5f };
	float sigma[SGNUM] = { 1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f };
	// float sigmas[4] = {6.5f, 8.5f, 10.5f, 12.5f};
	// float sigmas[4] = {3.5f, 5.5f, 7.5f, 9.5f};
	// float sigma[4] = {1.5f, 3.5f, 5.5f, 7.5f};
	float alpha = 20.0f;
	float beta = 1.0f;
	float gamma = 4e10;

	Image3D<float> vesselness_all( 
		Vec3i( image.get_size_x(),
               image.get_size_y()*SGNUM,
		       image.get_size_z())
	);
	
	for( int i=0; i<SGNUM; i++ ) { 
		cout << "Calculating Vesselness with KernelSize = " << 6*sigma[i] + 1;
		cout << " Sigma = " << sigma[i] << ". Please wait..." << endl;

		Image3D<Vesselness_Nor> vesselness;
		bool flag = VesselDetector::hessien( image, vesselness, 0, sigma[i], alpha, beta, gamma );

		smart_return_value( flag, "Hessien failed.", false );

		/*Image3D<float> vesselness2;
		for( int d=3; d>=0; --d ) { 
			vesselness.copyDimTo( vesselness2, d );
			vesselness2.show("Vesselness", 50);
		}*/

		int x, y, z;
		for(z=0;z<vesselness.get_size_z();z++) {
			for(y=20;y<vesselness.get_size_y()-20;y++) {
				for(x=0;x<vesselness.get_size_x();x++) {
					// vesselness_all.at(x, y+i*vesselness.get_size_y(), z) = vesselness.at(x,y,z)[0];
				}
			}
		}
	}
	vesselness_all.show("Vesselness", 50);

	// vesselness_all.saveVideo( "3d tubes vesselness.avi" );

	return true; 
}


bool Hessian_3D_Real_Data(void){
	Image3D<short> image_data;
	Image3D<Vesselness_Nor> vesselness;
	
	image_data.loadROI( "roi13.data" );
	Vec3i data_size = Vec3i(238, 223, 481);

	float alpha = 20.0f;
	float beta = 1.0f;
	float gamma = 4e10;
	float sigmas[3] = { 1.5, 3, 4.5 };
	Image3D<float> im_dispay( Vec3i(238*4, 223, 481) );
	for( int i=0; i<3; i++ ) {
		int ksize = int( (sigmas[i] - 0.35f) / 0.15f );
		if( ksize%2==0 ) ksize++;
		cout << "computing vesselness for ksize = " << ksize << endl;
		VesselDetector::hessien( image_data.getROI(), vesselness, ksize, 0, alpha, beta, gamma );
		int x, y, z;
		for( z=30; z<vesselness.get_size_z()-30; z++ ) {
			for( y=30; y<vesselness.get_size_y()-30; y++ ) {
				for( x=30; x<vesselness.get_size_x()-30; x++ ) {
					// im_dispay.at(x + i * data_size[0], y, z) = vesselness.at(x, y, z)[0];
				}
			}
		}
	}
	// im_dispay.show("Vesselness Response", 100);
	im_dispay.save("vesselness s 1_5 3 4_5.float.data", false, true);
	return 0;
}
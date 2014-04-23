#pragma once

#include "Data3D.h" 
#include <random> 

namespace SyntheticData {
	void Doughout( Data3D<short>& im_short ) {
		static int SX = 30; 
		static int SY = 30; 
		static int SZ = 30; 
		static double r1 = 5.5; // SX * 0.3; 
		static double r2 = 0.9; 

		im_short.reset( Vec3i(SX, SY, SZ) ); 

		// center of the doughout
		float cx = (float) SX / 2;
		float cy = (float) SY / 2;
		float cz = (float) SZ / 2 + 0.5f; 

		for( int z=0; z<im_short.SZ(); z++ ) {
			float dz = z - cz; // difference in z
			for( int y=0; y<SY; y++ ) for( int x=0; x<SX; x++ ) { 
				// difference in x
				float dx = x - cx; 
				// difference in y
				float dy = y - cy; 

				// distance to the center of the doughout
				double dc = sqrt( 1.0*dx*dx + 1.0*dy*dy );
				
				if( (dc-r1)*(dc-r1) + dz * dz < r2 * r2 ) {
					im_short.at(x,y,z) = 10000; 
				}
			}
		}
	}

	void Stick( Data3D<short>& im_short ) {
		im_short.reset( Vec3i(20,20,20) ); 
		for( int i=10; i<15; i++ ) {
			im_short.at(i,  i,  i)   = 10000; 
			im_short.at(i,  i,  i+1) = 10000; 
			im_short.at(i,  i+1,i)   = 10000; 
			im_short.at(i+1,i,  i)   = 10000; 
			im_short.at(i+1,i+1,i+1) = 10000; 
		}
	}
}
#pragma once

//#include "Graph.h"
//#include "MinSpanTree.h"
//#include "MinSpanTreeWrapper.h"
//#include "DataTypes.h"

#include <windows.h>		// Header File For Windows
#include <queue>

#include "GLViewer.h"


#include "glew.h"
#include <gl\gl.h>			// Header File For The OpenGL32 Library
#include <gl\glu.h>			// Header File For The GLu32 Library


#include "GLCamera.h" 

#include <vector>
#include <iostream>

namespace GLViewer
{
	// rendering object with Maximum Intenstiy Projection
	class Volumn : public GLViewer::Object {
	public:
		bool isMIP; // Using Maximum Intensity Projection
		/////////////////////////////////////////
		// Data
		///////////////////////
		// 3D Volumn Texture
		GLuint texture;
		// size of texture
		int texture_sx;
		int texture_sy;
		int texture_sz;
		// Original Data
		int sx;
		int sy;
		int sz;
		unsigned char* data;
		friend class VolumnWithROI;
	public:
		Volumn( unsigned char* im_data, const int& im_x, const int& im_y, const int& im_z ) {
			// From wikipedia: Do not forget that all 3 dimensions must be a power of 2! so your 2D textures must have 
			// the same size and this size be a power of 2 AND the number of layers (=2D textures) you use to create 
			// your 3D texture must be a power of 2 too.
			sx = im_x;
			sy = im_y;
			sz = im_z;

			static const double log2 = log(2.0);
			texture_sx = (int) pow(2, ceil( log( 1.0*sx )/log2 ));
			texture_sy = (int) pow(2, ceil( log( 1.0*sy )/log2 ));
			texture_sz = (int) pow(2, ceil( log( 1.0*sz )/log2 ));

			int texture_size = texture_sx * texture_sy * texture_sz;
			data = new (std::nothrow) unsigned char [ texture_size ];
			if( data==NULL ) {
				std::cout << "Unable to allocate memory for OpenGL rendering" << std::endl; return;
			}

			memset( data, 0, sizeof(unsigned char) * texture_size );
			for( int z=0;z<sz;z++ ) for( int y=0;y<sy;y++ ) for( int x=0; x<sx; x++ ) {
				data[ z*texture_sy*texture_sx + y*texture_sx + x] = im_data[ z*sy*sx + y*sx + x];
			}

			isMIP = true;
		}



		~Volumn() {
			if(data){
				delete[] data;
				data = NULL;
			}
		}

		void init(void){
			// Creating Textures
			glGenTextures(1, &texture); // Create The Texture
			glBindTexture(GL_TEXTURE_3D, texture);
			// Yuchen [Important]: For the following line of code
			// If the graphic hard do not have enough memory for the 3D texture,
			// OpenGL will fail to render the textures. However, since it is hardware
			// related, the porgramme won't show any assertions here. I may need to fix 
			// this issue later. But now, it has no problem rendering 3D texture with a 
			// size of 1024 * 1024 * 1024. 
			glTexImage3D(GL_TEXTURE_3D, 0, GL_LUMINANCE, 
				texture_sx, texture_sy, texture_sz, 0,
				GL_LUMINANCE, GL_UNSIGNED_BYTE, data );
			if(data){
				delete[] data;
				data = NULL;
			}
			
			////////////////////////////////////////
			//// Set up OpenGL
			//
			glEnable(GL_BLEND);
			glBlendFunc(GL_ONE, GL_ONE);
			glBlendEquationEXT( GL_MAX_EXT ); // Enable Blending For Maximum Intensity Projection

			// Use GL_NEAREST to see the voxels
			glEnable( GL_TEXTURE_3D ); // Enable Texture Mapping
			glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_MAG_FILTER, GL_LINEAR ); 
			glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
			// Sets the wrap parameter for texture coordinate
			glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE  ); 
			glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE  );
			glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_WRAP_R, GL_CLAMP_TO_EDGE  );

			glEnable( GL_POLYGON_SMOOTH_HINT );
			glHint (GL_POLYGON_SMOOTH_HINT, GL_NICEST);
			glHint (GL_PERSPECTIVE_CORRECTION_HINT, GL_NICEST);

			//glBindTexture(GL_TEXTURE_3D, NULL);
			//glDisable( GL_TEXTURE_3D ); 
			GLenum error = glGetError();

		}

		virtual void keyboard(unsigned char key ) {
			isMIP = !isMIP; 
		}

		struct Vec3f{
			float x, y, z;
			Vec3f( float x=0, float y=0, float z=0 ) : x(x), y(y), z(z) { }
			inline float dot( const Vec3f& v ) const { return x * v.x + y * v.y + z * v.z; }
			inline Vec3f cross( const Vec3f& v ) const {
				Vec3f res;
				res.x = y * v.z - z * v.y; 
				res.y = z * v.x - x * v.z; 
				res.z = x * v.y - y * v.x; 
				return res;
			}
			inline Vec3f operator-( const Vec3f& v) const { return Vec3f( x-v.x, y-v.y, z-v.z); }
			inline Vec3f operator+( const Vec3f& v) const { return Vec3f( x+v.x, y+v.y, z+v.z); }
			inline Vec3f operator*( const float& v) const { return Vec3f( v*x, v*y, v*z );      }
			inline float length() const { return sqrt(x*x + y*y + z*z); }
		}; 

		std::vector<Vec3f> intersectPoints( const Vec3f& center, const Vec3f& norm )
		{
			float t;
			std::vector<Vec3f> result;
			if( abs(norm.z) > 1.0e-3 ) {
				// (0, 0, t)
				t = center.dot(norm);
				t /= norm.z;
				if( t>=0 && t<=sz ) result.push_back( Vec3f(0,0,t) );
				// (0, sy, t)
				t = center.dot(norm)- norm.y * sy;
				t /= norm.z;
				if( t>=0 && t<=sz ) result.push_back( Vec3f(0,(float)sy,t) );
				// (sx, 0, t)
				t = center.dot(norm) - norm.x * sx;
				t /= norm.z;
				if( t>=0 && t<=sz ) result.push_back( Vec3f((float)sx,0,t) );
				// (sx, sy, t)
				t = center.dot(norm) - norm.y * sy - norm.x * sx;
				t /= norm.z;
				if( t>=0 && t<=sz ) result.push_back( Vec3f((float)sx,(float)sy,t) );
			}

			if( abs(norm.y) > 1.0e-3 ) {
				// (0, t, 0)
				t = center.dot(norm);
				t /= norm.y;
				if( t>=0 && t<=sy ) result.push_back( Vec3f(0,t,0) );
				// (sx, t, 0)
				t = center.dot(norm) - norm.x * sx;
				t /= norm.y;
				if( t>=0 && t<=sy ) result.push_back( Vec3f((float)sx,t,0) );
				// (0, t, sz)
				t = center.dot(norm) - norm.z * sz;
				t /= norm.y;
				if( t>=0 && t<=sy ) result.push_back( Vec3f(0,t,(float)sz) );
				// (sx, t, sz)
				t = center.dot(norm) - norm.z * sz - norm.x * sx;
				t /= norm.y;
				if( t>=0 && t<=sy ) result.push_back( Vec3f((float)sx,t,(float)sz) );
			}

			if( abs(norm.x) > 1.0e-3 ) {
				// (t, 0, 0)
				t = center.dot(norm);
				t /= norm.x;
				if( t>=0 && t<=sx ) result.push_back( Vec3f(t,0,0) );
				// (t, sy, 0)
				t = center.dot(norm) - norm.y * sy;
				t /= norm.x;
				if( t>=0 && t<=sx ) result.push_back( Vec3f(t,(float)sy,0) );
				// (t, 0, sz)
				t = center.dot(norm) - norm.z * sz;
				t /= norm.x;
				if( t>=0 && t<=sx ) result.push_back( Vec3f(t,0,(float)sz) );
				// (t, sy, sz)
				t = center.dot(norm) - norm.y * sy - norm.z * sz;
				t /= norm.x;
				if( t>=0 && t<=sx ) result.push_back( Vec3f(t,(float)sy,(float)sz) );
			}

			
			if( result.size()<=2 ) {
				result.clear();
			} else if( result.size()==3 ) {
				// don't need to do anything
			} else if( result.size()<=6 ) {
				// sort them based on signed angle:
				// http://stackoverflow.com/questions/20387282/compute-the-cross-section-of-a-cube

				Vec3f centroid(0,0,0);
				for( unsigned int i=0; i<result.size(); i++ ) {
					centroid.x += result[i].x;
					centroid.y += result[i].y;
					centroid.z += result[i].z;
				}
				centroid.x /= result.size(); 
				centroid.y /= result.size(); 
				centroid.z /= result.size(); 

				// We are not using the first index
				static float angles[6];
				static Vec3f va[6];
				for( unsigned int i=0; i<result.size(); i++ ) {
					va[i] = result[i] - centroid;
					float dotproduct = va[0].dot( va[i] )/( va[i].length()*va[0].length() );
					dotproduct = max( -1, min( 1, dotproduct) );
					angles[i] = acos( dotproduct );
					if( abs( angles[i] ) < 1e-3 ) continue;

					Vec3f cross = va[0].cross( va[i] );
					if( cross.dot( norm ) < 0 ) {
						angles[i] = -angles[i];
					}
				}
				for( unsigned int i=0; i<result.size(); i++ ) {
					for( unsigned int j=i+1; j<result.size(); j++ ) {
						if( angles[i] < angles[j] ) {
							std::swap( angles[i], angles[j] );
							std::swap( result[i], result[j] );
						}
					}
				}
			} else {
				std::cout << "Error (Volumn.h): There are at most six points" << std::endl;
			}
			return result;
		}

		void render(void){
			static char ONE = 0x1;
			static char TWO = 0x2;
			static char THREE = 0x4;

			GLenum error = glGetError();

			if( isMIP ) {
				// visualizing the data with maximum intensity projection
				glBindTexture(GL_TEXTURE_3D, texture);
				glBegin(GL_QUADS);
				glColor3f( 1.0f, 1.0f, 1.0f );
				for( float i=0; i<=sz; i+=0.27f ) {
					glTexCoord3f( 0.0f,               0.0f,               1.0f*i/texture_sz ); glVertex3f( 0.0f,    0.0f,    i );
					glTexCoord3f( 1.0f*sx/texture_sx, 0.0f,               1.0f*i/texture_sz ); glVertex3f( 1.0f*sx, 0.0f,    i );
					glTexCoord3f( 1.0f*sx/texture_sx, 1.0f*sy/texture_sy, 1.0f*i/texture_sz ); glVertex3f( 1.0f*sx, 1.0f*sy, i );
					glTexCoord3f( 0.0f,               1.0f*sy/texture_sy, 1.0f*i/texture_sz ); glVertex3f( 0.0f,    1.0f*sy, i );
				}
				//for( float i=0; i<=sy; i+=0.27f ) {
				//	glTexCoord3f( 0.0f,               1.0f*i/texture_sy, 0.0f );               glVertex3f( 0.0f,    i, 0.0f );
				//	glTexCoord3f( 1.0f*sx/texture_sx, 1.0f*i/texture_sy, 0.0f );               glVertex3f( 1.0f*sx, i, 0.0f );
				//	glTexCoord3f( 1.0f*sx/texture_sx, 1.0f*i/texture_sy, 1.0f*sz/texture_sz ); glVertex3f( 1.0f*sx, i, 1.0f*sz );
				//	glTexCoord3f( 0.0f,               1.0f*i/texture_sy, 1.0f*sz/texture_sz ); glVertex3f( 0.0f,    i, 1.0f*sz );
				//}
				//for( float i=0; i<=sx; i+=0.27f ) {
				//	glTexCoord3f( 1.0f*i/texture_sx, 0.0f,               0.0f );               glVertex3f( i, 0.0f,    0.0f );
				//	glTexCoord3f( 1.0f*i/texture_sx, 1.0f*sy/texture_sy, 0.0f );               glVertex3f( i, 1.0f*sy, 0.0f );
				//	glTexCoord3f( 1.0f*i/texture_sx, 1.0f*sy/texture_sy, 1.0f*sz/texture_sz ); glVertex3f( i, 1.0f*sy, 1.0f*sz );
				//	glTexCoord3f( 1.0f*i/texture_sx, 0.0f,               1.0f*sz/texture_sz ); glVertex3f( i, 0.0f,    1.0f*sz );
				//}
				glEnd();
				glBindTexture( GL_TEXTURE_3D, NULL );

				GLenum error = glGetError();
			} 
			else 
			{
				Vec3f center, vz;
				center.x = cam.t[0]; center.y = cam.t[1]; center.z = cam.t[2];
				vz.x = cam.vec_x[1]*cam.vec_y[2] - cam.vec_x[2]*cam.vec_y[1]; 
				vz.y = cam.vec_x[2]*cam.vec_y[0] - cam.vec_x[0]*cam.vec_y[2]; 
				vz.z = cam.vec_x[0]*cam.vec_y[1] - cam.vec_x[1]*cam.vec_y[0]; 
				std::vector<Vec3f> points = intersectPoints( center, vz );

				glColor3f( 1.0f, 1.0f, 1.0f );
				glBindTexture(GL_TEXTURE_3D, texture);
				glBegin( GL_TRIANGLE_FAN );
				for( unsigned int i=0; i<points.size(); i++ ) {
					glTexCoord3f(
						points[i].x / texture_sx,
						points[i].y / texture_sy,
						points[i].z / texture_sz );
					glVertex3f( points[i].x, points[i].y, points[i].z ); 
				}
				glEnd();
				glBindTexture( GL_TEXTURE_3D, NULL );

				// draw the boundary of the cross section
				glColor3f( 0.3f, 0.3f, 0.3f );
				glBegin( GL_LINE_LOOP );
				for( unsigned int i=0; i<points.size(); i++ ) {
					glVertex3f( points[i].x, points[i].y, points[i].z ); 
				}
				glEnd();

				// draw the boundary of the box
				glColor3f( 0.0f, 0.0f, 0.8f );
				// left borders
				glBegin(GL_LINE_LOOP);
				glVertex3i( 0,0,0 );
				glVertex3i( 0,0,sz );
				glVertex3i( 0,sy,sz );
				glVertex3i( 0,sy,0 );
				glEnd();
				// right borders
				glBegin(GL_LINE_LOOP);
				glVertex3i( sx,0,0 );
				glVertex3i( sx,0,sz );
				glVertex3i( sx,sy,sz );
				glVertex3i( sx,sy,0 );
				glEnd();
				// parrenl lines to x
				glBegin(GL_LINES);
				glVertex3i( 0,0,0 );  glVertex3i( sx,0,0 );
				glVertex3i( 0,0,sz ); glVertex3i( sx,0,sz );
				glVertex3i( 0,sy,sz );glVertex3i( sx,sy,sz );
				glVertex3i( 0,sy,0 ); glVertex3i( sx,sy,0 );
				glEnd();
			}
		}

		unsigned int size_x() const { return sx; }
		unsigned int size_y() const { return sy; }
		unsigned int size_z() const { return sz; }
	}; 

	//class VolumnWithROI : public Volumn {
	//public:
	//	int xmin, ymin, zmin;
	//	int xmax, ymax, zmax;

	//	VolumnWithROI( unsigned char* im_data, const int& im_x, const int& im_y, const int& im_z,
	//		int xmin, int ymin, int zmin, int xmax, int ymax, int zmax ) 
	//		: Volumn( im_data, im_x, im_y, im_z )
	//		, xmin( xmin )
	//		, ymin( ymin )
	//		, zmin( zmin )
	//		, xmax( xmax )
	//		, ymax( ymax )
	//		, zmax( zmax )
	//	{

	//	}

	//	void render(void){
	//		
	//		// visualizing the data with maximum intensity projection
	//		glBindTexture(GL_TEXTURE_3D, texture);
	//		glBegin(GL_QUADS);
	//		glColor3f( 0.5f, 0.5f, 0.5f );
	//		for( float i=0; i<=sz; i+=1.0f ) {
	//			glTexCoord3f( 0.0f,               0.0f,               1.0f*i/texture_sz ); glVertex3f( 0.0f,    0.0f,    i );
	//			glTexCoord3f( 1.0f*sx/texture_sx, 0.0f,               1.0f*i/texture_sz ); glVertex3f( 1.0f*sx, 0.0f,    i );
	//			glTexCoord3f( 1.0f*sx/texture_sx, 1.0f*sy/texture_sy, 1.0f*i/texture_sz ); glVertex3f( 1.0f*sx, 1.0f*sy, i );
	//			glTexCoord3f( 0.0f,               1.0f*sy/texture_sy, 1.0f*i/texture_sz ); glVertex3f( 0.0f,    1.0f*sy, i );
	//		}
	//		for( float i=0; i<=sy; i+=1.0f ) {
	//			glTexCoord3f( 0.0f,               1.0f*i/texture_sy, 0.0f );               glVertex3f( 0.0f,    i, 0.0f );
	//			glTexCoord3f( 1.0f*sx/texture_sx, 1.0f*i/texture_sy, 0.0f );               glVertex3f( 1.0f*sx, i, 0.0f );
	//			glTexCoord3f( 1.0f*sx/texture_sx, 1.0f*i/texture_sy, 1.0f*sz/texture_sz ); glVertex3f( 1.0f*sx, i, 1.0f*sz );
	//			glTexCoord3f( 0.0f,               1.0f*i/texture_sy, 1.0f*sz/texture_sz ); glVertex3f( 0.0f,    i, 1.0f*sz );
	//		}
	//		for( float i=0; i<=sx; i+=1.0f ) {
	//			glTexCoord3f( 1.0f*i/texture_sx, 0.0f,               0.0f );               glVertex3f( i, 0.0f,    0.0f );
	//			glTexCoord3f( 1.0f*i/texture_sx, 1.0f*sy/texture_sy, 0.0f );               glVertex3f( i, 1.0f*sy, 0.0f );
	//			glTexCoord3f( 1.0f*i/texture_sx, 1.0f*sy/texture_sy, 1.0f*sz/texture_sz ); glVertex3f( i, 1.0f*sy, 1.0f*sz );
	//			glTexCoord3f( 1.0f*i/texture_sx, 0.0f,               1.0f*sz/texture_sz ); glVertex3f( i, 0.0f,    1.0f*sz );
	//		}
	//		// the roi
	//		glColor3f( 1.0f, 1.0f, 0.0f );
	//		for( float i=zmin; i<=zmax; i+=1.0f ) {
	//			glTexCoord3f( 1.0f*xmin/texture_sx, 1.0f*ymin/texture_sx, 1.0f*i/texture_sz ); glVertex3f( 1.0f*xmin, 1.0f*ymin, i );
	//			glTexCoord3f( 1.0f*xmax/texture_sx, 1.0f*ymin/texture_sx, 1.0f*i/texture_sz ); glVertex3f( 1.0f*xmax, 1.0f*ymin, i );
	//			glTexCoord3f( 1.0f*xmax/texture_sx, 1.0f*ymax/texture_sy, 1.0f*i/texture_sz ); glVertex3f( 1.0f*xmax, 1.0f*ymax, i );
	//			glTexCoord3f( 1.0f*xmin/texture_sx, 1.0f*ymax/texture_sy, 1.0f*i/texture_sz ); glVertex3f( 1.0f*xmin, 1.0f*ymax, i );
	//		}
	//		//for( float i=ymin; i<=ymax; i+=1.0f ) {
	//		//	glTexCoord3f( 0.0f,               1.0f*i/texture_sy, 0.0f );               glVertex3f( 0.0f,    i, 0.0f );
	//		//	glTexCoord3f( 1.0f*sx/texture_sx, 1.0f*i/texture_sy, 0.0f );               glVertex3f( 1.0f*sx, i, 0.0f );
	//		//	glTexCoord3f( 1.0f*sx/texture_sx, 1.0f*i/texture_sy, 1.0f*sz/texture_sz ); glVertex3f( 1.0f*sx, i, 1.0f*sz );
	//		//	glTexCoord3f( 0.0f,               1.0f*i/texture_sy, 1.0f*sz/texture_sz ); glVertex3f( 0.0f,    i, 1.0f*sz );
	//		//}
	//		//for( float i=zmin; i<=zmax; i+=1.0f ) {
	//		//	glTexCoord3f( 1.0f*i/texture_sx, 0.0f,               0.0f );               glVertex3f( i, 0.0f,    0.0f );
	//		//	glTexCoord3f( 1.0f*i/texture_sx, 1.0f*sy/texture_sy, 0.0f );               glVertex3f( i, 1.0f*sy, 0.0f );
	//		//	glTexCoord3f( 1.0f*i/texture_sx, 1.0f*sy/texture_sy, 1.0f*sz/texture_sz ); glVertex3f( i, 1.0f*sy, 1.0f*sz );
	//		//	glTexCoord3f( 1.0f*i/texture_sx, 0.0f,               1.0f*sz/texture_sz ); glVertex3f( i, 0.0f,    1.0f*sz );
	//		//}
	//		glEnd();
	//		glBindTexture( GL_TEXTURE_3D, NULL );
	//	}
	//}; 
}
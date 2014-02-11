#pragma once
#ifndef _CRT_SECURE_NO_DEPRECATE
	#define _CRT_SECURE_NO_DEPRECATE
#endif

#include "Graph.h"
#include "MinSpanTree.h"
#include "MinSpanTreeWrapper.h"
#include "DataTypes.h"

#include <windows.h>		// Header File For Windows
#include <queue>

#include "GLViewer.h"
#include "gl\glew.h"
#include <gl\gl.h>			// Header File For The OpenGL32 Library
#include <gl\glu.h>			// Header File For The GLu32 Library


#include "GLCamera.h" 

namespace GLViewer
{
	// rendering object with Maximum Intenstiy Projection
	class Volumn : public GLViewer::Object {
		friend class VolumnWithROI;
		// rendeing mode
	public:
		enum RenderMode{ 
			MIP, //Maximum Intensity Projection
			CrossSection, 
			Surface
		} render_mode; 
	private:
		/////////////////////////////////////////
		// Data
		///////////////////////
		// 3D Volumn Texture
		GLuint texture;
		// size of texture
		int texture_sx, texture_sy, texture_sz;
		// Original Data
		int sx, sy, sz; 
		// Texture Data
		unsigned char* data; 
		// Reference to the camera
		GLCamera* ptrCam; 
	public:
		Volumn(unsigned char* im_data, const int& im_x, const int& im_y, const int& im_z,  GLCamera* ptrCamera = NULL ) 
			: sx( im_x )
			, sy( im_y )
			, sz( im_z )
			, ptrCam( ptrCamera )
		{
			// From wikipedia: Do not forget that all 3 dimensions must be a power of 2! so your 2D textures must have 
			// the same size and this size be a power of 2 AND the number of layers (=2D textures) you use to create 
			// your 3D texture must be a power of 2 too.
			static const double log2 = log(2.0);
			texture_sx = (int) pow(2, ceil( log( 1.0*sx )/log2 ));
			texture_sy = (int) pow(2, ceil( log( 1.0*sy )/log2 ));
			texture_sz = (int) pow(2, ceil( log( 1.0*sz )/log2 ));

			// allocating memeory for texture
			int texture_size = texture_sx * texture_sy * texture_sz;
			data = new (nothrow) unsigned char [ texture_size ];
			if( data==NULL ) {
				cout << "Unable to allocate memory for OpenGL texture" << endl;
				return;
			}

			memset( data, 0, sizeof(unsigned char) * texture_size );
			for( int z=0;z<sz;z++ ) for( int y=0;y<sy;y++ ) for( int x=0; x<sx; x++ ) {
				data[ z*texture_sy*texture_sx + y*texture_sx + x] = im_data[ z*sy*sx + y*sx + x];
			}

			render_mode = MIP;
		}


		bool update_data( unsigned char* im_data ) {
			if( data==NULL ) {
				return false; 
			}
			// update data for texture
			int texture_size = texture_sx * texture_sy * texture_sz; 
			for( int z=0;z<sz;z++ ) for( int y=0;y<sy;y++ ) { 
				memcpy( data + z*texture_sy*texture_sx + y*texture_sx, im_data + z*sy*sx + y*sx, sx );
			} 
			return true;
		}

		bool update_texture(  ) {
			if( data==NULL ) {
				return false; 
			}
			// update texture for rendering 
			glBindTexture(GL_TEXTURE_3D, texture);
			glTexImage3D(GL_TEXTURE_3D, 0, GL_LUMINANCE, 
				texture_sx, texture_sy, texture_sz, 0,
				GL_LUMINANCE, GL_UNSIGNED_BYTE, data );
			return true;
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

			//////////////////////////////////////
			// Set up OpenGL
			
			// Enable Blending For Maximum Intensity Projection
			setRenderMode( render_mode );
			
			// Antialiasing
			glEnable (GL_LINE_SMOOTH);
			glHint (GL_LINE_SMOOTH_HINT, GL_NICEST );

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
		}

		void setRenderMode( RenderMode mode ){
			switch( mode ) {
			case MIP:
				glDisable(GL_DEPTH_TEST);
				glEnable(GL_BLEND);
				glBlendFunc(GL_ONE, GL_ONE);
				glBlendEquation( GL_MAX_EXT ); 
				cout << "Volumn Rendeing Mode is set to MIP" << endl;
				break;
			case CrossSection:
				glEnable(GL_DEPTH_TEST);
				glDisable(GL_BLEND);
				cout << "Volumn Rendeing Mode is set to CrossSection" << endl;
				break;
			case Surface:
				glEnable(GL_DEPTH_TEST);
				glDisable(GL_BLEND);
				cout << "Volumn Rendeing Mode is set to Surface" << endl;
				break;
			}
			render_mode = mode; 
		}

		virtual void keyboard( unsigned char key ) {
			switch (render_mode){
			case MIP:                setRenderMode( CrossSection ); break;
			case CrossSection:       setRenderMode( Surface ); break;
			case Surface:            setRenderMode( MIP ); break;
			}
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
			inline Vec3f& operator+=( const Vec3f& v) {
				this->x += v.x;
				this->y += v.y;
				this->z += v.z;
				return *this;
			}
			template<class T>
			inline Vec3f& operator/=( const T& v) { 
				this->x /= v;
				this->y /= v;
				this->z /= v;
				return *this; 
			}
			inline float length() const { return sqrt(x*x + y*y + z*z); }
		}; 

		vector<Vec3f> intersectPoints( const Vec3f& center, const Vec3f& norm )
		{
			float t;
			vector<Vec3f> result;
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
					centroid += result[i];
				}
				centroid /= result.size(); 

				// We are not using the first index
				static float signed_angle[6];
				for( unsigned int i=0; i<result.size(); i++ ) {
					static Vec3f va[6];
					va[i] = result[i] - centroid;
					float dotproduct = va[0].dot( va[i] )/( va[i].length()*va[0].length() );
					// constraint the result of dotproduct be within -1 and 1 (it might
					// sometime not with this range only because of floating point 
					// calculation accuracy )
					if( dotproduct<-1 ){
						dotproduct = -1;
					} else if( dotproduct>1 ) {
						dotproduct = 1;
					}
					signed_angle[i] = acos( dotproduct );
					if( abs( signed_angle[i] ) < 1e-3 ) continue;

					Vec3f cross = va[0].cross( va[i] );
					if( cross.dot( norm ) < 0 ) {
						signed_angle[i] = -signed_angle[i];
					}
				}
				// bubble sort the result by signed_angle
				for( unsigned int i=0; i<result.size(); i++ ) {
					for( unsigned int j=i+1; j<result.size(); j++ ) {
						if( signed_angle[i] < signed_angle[j] ) {
							std::swap( signed_angle[i], signed_angle[j] );
							std::swap( result[i], result[j] );
						}
					}
				}
			} else {
				cout << "Error (Volumn.h): There are at most six points" << endl;
			}
			return result;
		}


		void render_volumn( const float& x_increase = 1.0f, const float& y_increase = 1.0f, const float& z_increase = 1.0f ){
			glBindTexture(GL_TEXTURE_3D, texture);
			glBegin(GL_QUADS);
			for( float i=0.0f; i<=sx-1; i+=x_increase ) {
				glTexCoord3f( 1.0f*(i+0.5f)/texture_sx, 0.0f,                      0.0f );                      glVertex3f( i, 0.0f,        0.0f );
				glTexCoord3f( 1.0f*(i+0.5f)/texture_sx, 1.0f*(sy-0.5f)/texture_sy, 0.0f );                      glVertex3f( i, 1.0f*(sy-1), 0.0f );
				glTexCoord3f( 1.0f*(i+0.5f)/texture_sx, 1.0f*(sy-0.5f)/texture_sy, 1.0f*(sz-0.5f)/texture_sz ); glVertex3f( i, 1.0f*(sy-1), 1.0f*(sz-1) );
				glTexCoord3f( 1.0f*(i+0.5f)/texture_sx, 0.0f,                      1.0f*(sz-0.5f)/texture_sz ); glVertex3f( i, 0.0f,        1.0f*(sz-1) );
			}
			for( float i=0.0f; i<=sy-1; i+=y_increase ) {
				glTexCoord3f( 0.0f,                      1.0f*(i+0.5f)/texture_sy, 0.0f );                      glVertex3f( 0.0f,        i, 0.0f );
				glTexCoord3f( 1.0f*(sx-0.5f)/texture_sx, 1.0f*(i+0.5f)/texture_sy, 0.0f );                      glVertex3f( 1.0f*(sx-1), i, 0.0f );
				glTexCoord3f( 1.0f*(sx-0.5f)/texture_sx, 1.0f*(i+0.5f)/texture_sy, 1.0f*(sz-0.5f)/texture_sz ); glVertex3f( 1.0f*(sx-1), i, 1.0f*(sz-1) );
				glTexCoord3f( 0.0f,                      1.0f*(i+0.5f)/texture_sy, 1.0f*(sz-0.5f)/texture_sz ); glVertex3f( 0.0f,        i, 1.0f*(sz-1) );
			}
			for( float i=0.0f; i<=sz-1; i+=z_increase ) {
				glTexCoord3f( 0.0f,                      0.0f,                      1.0f*(i+0.5f)/texture_sz ); glVertex3f( 0.0f,        0.0f,        i );
				glTexCoord3f( 1.0f*(sx-0.5f)/texture_sx, 0.0f,                      1.0f*(i+0.5f)/texture_sz ); glVertex3f( 1.0f*(sx-1), 0.0f,        i );
				glTexCoord3f( 1.0f*(sx-0.5f)/texture_sx, 1.0f*(sy-0.5f)/texture_sy, 1.0f*(i+0.5f)/texture_sz ); glVertex3f( 1.0f*(sx-1), 1.0f*(sy-1), i );
				glTexCoord3f( 0.0f,                      1.0f*(sy-0.5f)/texture_sy, 1.0f*(i+0.5f)/texture_sz ); glVertex3f( 0.0f,        1.0f*(sy-1), i );
			}
			glEnd();
			glBindTexture( GL_TEXTURE_3D, NULL );
		}

		void render_outline(void){
			float x_min = -0.5f;
			float y_min = -0.5f;
			float z_min = -0.5f;
			float X_MAX = sx-1 - x_min;
			float Y_MAX = sy-1 - y_min;
			float Z_MAX = sz-1 - z_min;
			// left borders
			glBegin(GL_LINE_LOOP);
			glVertex3f( x_min, y_min, z_min );
			glVertex3f( x_min, Y_MAX, z_min );
			glVertex3f( x_min, Y_MAX, Z_MAX );
			glVertex3f( x_min, y_min, Z_MAX );
			glEnd();
			// right borders
			glBegin(GL_LINE_LOOP);
			glVertex3f( X_MAX, y_min, z_min );
			glVertex3f( X_MAX, Y_MAX, z_min );
			glVertex3f( X_MAX, Y_MAX, Z_MAX );
			glVertex3f( X_MAX, y_min, Z_MAX );
			glEnd();
			// parrenl lines to x-axix
			glBegin(GL_LINES);
			glVertex3f( x_min, y_min, z_min ); glVertex3f( X_MAX, y_min, z_min );
			glVertex3f( x_min, Y_MAX, z_min ); glVertex3f( X_MAX, Y_MAX, z_min );
			glVertex3f( x_min, Y_MAX, Z_MAX ); glVertex3f( X_MAX, Y_MAX, Z_MAX );
			glVertex3f( x_min, y_min, Z_MAX ); glVertex3f( X_MAX, y_min, Z_MAX );
			glEnd();
		}

		void render(void){
			static char ONE = 0x1;
			static char TWO = 0x2;
			static char THREE = 0x4;

			if( render_mode == MIP ) {
				// visualizing the data with maximum intensity projection
				glColor3f( 1.0f, 1.0f, 1.0f ); render_volumn();
				glColor3f( 0.2f, 0.2f, 0.2f ); render_outline();
			} 
			else if( render_mode == CrossSection ) // Yuchen: This rendering Mode requires the information of camera 
			{
				if( ptrCam==NULL ) return; 
				// retrive camera infomation 
				Vec3f center, vz;
				center.x = ptrCam->t[0];
				center.y = ptrCam->t[1];
				center.z = ptrCam->t[2];
				vz.x = ptrCam->vec_x[1]*ptrCam->vec_y[2] - ptrCam->vec_x[2]*ptrCam->vec_y[1]; 
				vz.y = ptrCam->vec_x[2]*ptrCam->vec_y[0] - ptrCam->vec_x[0]*ptrCam->vec_y[2]; 
				vz.z = ptrCam->vec_x[0]*ptrCam->vec_y[1] - ptrCam->vec_x[1]*ptrCam->vec_y[0]; 
				// get the cross section of cube
				// the cross section can vary from a point to hexagon 
				vector<Vec3f> points = intersectPoints( center, vz );

				glEnable(GL_DEPTH_TEST);

				// draw the cross section
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

				// draw the boundary of the box
				glColor3f( 0.0f, 0.0f, 0.8f );
				render_outline(); 

				// We want to boarder to be visible all the time; therefore disable depth_test
				glDisable(GL_DEPTH_TEST); 
				// draw the boundary of the cross section
				glColor3f( 0.3f, 0.3f, 0.3f );
				glBegin( GL_LINE_LOOP );
				for( unsigned int i=0; i<points.size(); i++ ) {
					glVertex3f( points[i].x, points[i].y, points[i].z ); 
				}
				glEnd();

			} else if ( render_mode==Surface ) {
				glColor3f( 1.0f, 1.0f, 1.0f ); render_volumn( sx-1.0f, sy-1.0f, sz-1.f );
				glColor3f( 0.2f, 0.2f, 0.2f ); render_outline();
			}
		} 

		unsigned int size_x() const { return sx; }
		unsigned int size_y() const { return sy; }
		unsigned int size_z() const { return sz; }
	}; 
}
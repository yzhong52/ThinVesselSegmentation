#pragma once

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

namespace GLViewer
{
	// rendering object with Maximum Intenstiy Projection
	class Volumn : public GLViewer::Object {
		char mode; // display mode
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
			data = new (nothrow) unsigned char [ texture_size ];
			if( data==NULL ) {
				cout << "Unable to allocate memory for OpenGL rendering" << endl; return;
			}

			memset( data, 0, sizeof(unsigned char) * texture_size );
			for( int z=0;z<sz;z++ ) for( int y=0;y<sy;y++ ) for( int x=0; x<sx; x++ ) {
				data[ z*texture_sy*texture_sx + y*texture_sx + x] = im_data[ z*sy*sx + y*sx + x];
			}

			mode = 0x1;
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

			//////////////////////////////////////
			// Set up OpenGL
			glEnable( GL_TEXTURE_3D ); // Enable Texture Mapping
			glBlendFunc(GL_ONE, GL_ONE);
			glEnable(GL_BLEND);
			glBlendEquationEXT( GL_MAX_EXT ); // Enable Blending For Maximum Intensity Projection

			// Use GL_NEAREST to see the voxels
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

		virtual void keyboard(unsigned char key ) {
			if ( key == '\t' ) { /*TAB key*/
				if( mode==0x01 ) {
					mode = 0x02;
				} else{
					mode = 0x01;
				}
			}
		}

		void render(void){
			static char ONE = 0x1;
			static char TWO = 0x2;
			static char THREE = 0x4;
			if( mode & ONE ) {
				glBindTexture(GL_TEXTURE_3D, texture);
				glBegin(GL_QUADS);
				for( float i=0; i<=sz; i+=1.25f ) {
					glTexCoord3f( 0.0f,               0.0f,               1.0f*i/texture_sz ); glVertex3f( 0.0f,    0.0f,    i );
					glTexCoord3f( 1.0f*sx/texture_sx, 0.0f,               1.0f*i/texture_sz ); glVertex3f( 1.0f*sx, 0.0f,    i );
					glTexCoord3f( 1.0f*sx/texture_sx, 1.0f*sy/texture_sy, 1.0f*i/texture_sz ); glVertex3f( 1.0f*sx, 1.0f*sy, i );
					glTexCoord3f( 0.0f,               1.0f*sy/texture_sy, 1.0f*i/texture_sz ); glVertex3f( 0.0f,    1.0f*sy, i );
				}
				for( float i=0; i<=sy; i+=1.25f ) {
					glTexCoord3f( 0.0f,               1.0f*i/texture_sy, 0.0f );               glVertex3f( 0.0f,    i, 0.0f );
					glTexCoord3f( 1.0f*sx/texture_sx, 1.0f*i/texture_sy, 0.0f );               glVertex3f( 1.0f*sx, i, 0.0f );
					glTexCoord3f( 1.0f*sx/texture_sx, 1.0f*i/texture_sy, 1.0f*sz/texture_sz ); glVertex3f( 1.0f*sx, i, 1.0f*sz );
					glTexCoord3f( 0.0f,               1.0f*i/texture_sy, 1.0f*sz/texture_sz ); glVertex3f( 0.0f,    i, 1.0f*sz );
				}
				for( float i=0; i<=sx; i+=1.25f ) {
					glTexCoord3f( 1.0f*i/texture_sx, 0.0f,               0.0f );               glVertex3f( i, 0.0f,    0.0f );
					glTexCoord3f( 1.0f*i/texture_sx, 1.0f*sy/texture_sy, 0.0f );               glVertex3f( i, 1.0f*sy, 0.0f );
					glTexCoord3f( 1.0f*i/texture_sx, 1.0f*sy/texture_sy, 1.0f*sz/texture_sz ); glVertex3f( i, 1.0f*sy, 1.0f*sz );
					glTexCoord3f( 1.0f*i/texture_sx, 0.0f,               1.0f*sz/texture_sz ); glVertex3f( i, 0.0f,    1.0f*sz );
				}
				glEnd();
				glBindTexture( GL_TEXTURE_3D, NULL );
			}


			if( mode & TWO && false ) {
				GLfloat vec_z[3];
				vec_z[0] = vec_x[1]*vec_y[2] - vec_x[2]*vec_y[1]; 
				vec_z[1] = vec_x[2]*vec_y[0] - vec_x[0]*vec_y[2]; 
				vec_z[2] = vec_x[0]*vec_y[1] - vec_x[1]*vec_y[0]; 
				glBindTexture(GL_TEXTURE_3D, texture);
				glBegin(GL_QUADS);
				for( int i=0; i<=0; i++ ) {
					// lower left
					glTexCoord3f(
						(t[0]-vec_x[0]*sx*2-vec_y[0]*sy*2-vec_z[0]*i)/texture_sx,
						(t[1]-vec_x[1]*sx*2-vec_y[1]*sy*2-vec_z[0]*i)/texture_sy,
						(t[2]-vec_x[2]*sx*2-vec_y[2]*sy*2-vec_z[0]*i)/texture_sz );
					glVertex3f( 
						(t[0]-vec_x[0]*sx*2-vec_y[0]*sy*2-vec_z[0]*i),
						(t[1]-vec_x[1]*sx*2-vec_y[1]*sy*2-vec_z[0]*i),
						(t[2]-vec_x[2]*sx*2-vec_y[2]*sy*2-vec_z[0]*i) );
					// lower right
					glTexCoord3f(
						(t[0]+vec_x[0]*sx*2-vec_y[0]*sy*2-vec_z[0]*i)/texture_sx,
						(t[1]+vec_x[1]*sx*2-vec_y[1]*sy*2-vec_z[0]*i)/texture_sy,
						(t[2]+vec_x[2]*sx*2-vec_y[2]*sy*2-vec_z[0]*i)/texture_sz );
					glVertex3f( 
						(t[0]+vec_x[0]*sx*2-vec_y[0]*sy*2-vec_z[0]*i),
						(t[1]+vec_x[1]*sx*2-vec_y[1]*sy*2-vec_z[0]*i),
						(t[2]+vec_x[2]*sx*2-vec_y[2]*sy*2-vec_z[0]*i) );
					// upper right
					glTexCoord3f(
						(t[0]+vec_x[0]*sx*2+vec_y[0]*sy*2-vec_z[0]*i)/texture_sx,
						(t[1]+vec_x[1]*sx*2+vec_y[1]*sy*2-vec_z[0]*i)/texture_sy,
						(t[2]+vec_x[2]*sx*2+vec_y[2]*sy*2-vec_z[0]*i)/texture_sz );
					glVertex3f( 
						(t[0]+vec_x[0]*sx*2+vec_y[0]*sy*2-vec_z[0]*i),
						(t[1]+vec_x[1]*sx*2+vec_y[1]*sy*2-vec_z[0]*i),
						(t[2]+vec_x[2]*sx*2+vec_y[2]*sy*2-vec_z[0]*i) );
					// uppper left
					glTexCoord3f(
						(t[0]-vec_x[0]*sx+vec_y[0]*sy-vec_z[0]*i)/texture_sx,
						(t[1]-vec_x[1]*sx+vec_y[1]*sy-vec_z[0]*i)/texture_sy,
						(t[2]-vec_x[2]*sx+vec_y[2]*sy-vec_z[0]*i)/texture_sz );
					glVertex3f( 
						(t[0]-vec_x[0]*sx+vec_y[0]*sy-vec_z[0]*i),
						(t[1]-vec_x[1]*sx+vec_y[1]*sy-vec_z[0]*i),
						(t[2]-vec_x[2]*sx+vec_y[2]*sy-vec_z[0]*i) );
				}
				glEnd();
				glBindTexture( GL_TEXTURE_3D, NULL );
			}
		}

		unsigned int size_x() const { return sx; }
		unsigned int size_y() const { return sy; }
		unsigned int size_z() const { return sz; }
	}; 
}
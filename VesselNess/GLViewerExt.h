#pragma once

#include "Graph.h"
#include "MinSpanTree.h"
#include "MinSpanTreeWrapper.h"
#include "DataTypes.h"

#include <windows.h>		// Header File For Windows
#include <queue>
#include <iostream>

// Using OpenCV ot Save the Scene to video
#define _CRT_SECURE_NO_DEPRECATE
#include <opencv\cv.h>      

#include "GLViewer.h"
#include "gl\glew.h"
#include <gl\gl.h>			// Header File For The OpenGL32 Library
#include <gl\glu.h>			// Header File For The GLu32 Library

namespace GLViewerExt
{

	class Volumn : public GLViewer::Object {
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
			memset( data, 0, sizeof(unsigned char) * texture_size );
			if( data==NULL ) {
				cout << "Unable to allocate memory for OpenGL rendering" << endl;
				return;
			}
			for( int z=0;z<sz;z++ ) for( int y=0;y<sy;y++ ) for( int x=0; x<sx; x++ ) {
				data[ z*texture_sy*texture_sx + y*texture_sx + x] = im_data[ z*sy*sx + y*sx + x];
			}	
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

		void render(void){
			glBindTexture(GL_TEXTURE_3D, texture);
			glBegin(GL_QUADS);
			for( int i=0; i<=sz; i++ ) {
				glTexCoord3f( 0,                  0,                  1.0f*i/texture_sz ); glVertex3i( 0,  0,  i );
				glTexCoord3f( 1.0f*sx/texture_sx, 0,                  1.0f*i/texture_sz ); glVertex3i( sx, 0,  i );
				glTexCoord3f( 1.0f*sx/texture_sx, 1.0f*sy/texture_sy, 1.0f*i/texture_sz ); glVertex3i( sx, sy, i );
				glTexCoord3f( 0,                  1.0f*sy/texture_sy, 1.0f*i/texture_sz ); glVertex3i( 0,  sy, i );
			}
			for( int i=0; i<=sy; i++ ) {
				glTexCoord3f( 0,                  1.0f*i/texture_sy, 0 );                  glVertex3i(  0, i,  0 );
				glTexCoord3f( 1.0f*sx/texture_sx, 1.0f*i/texture_sy, 0 );                  glVertex3i( sx, i,  0 );
				glTexCoord3f( 1.0f*sx/texture_sx, 1.0f*i/texture_sy, 1.0f*sz/texture_sz ); glVertex3i( sx, i, sz );
				glTexCoord3f( 0,                  1.0f*i/texture_sy, 1.0f*sz/texture_sz ); glVertex3i(  0, i, sz );
			}
			for( int i=0; i<=sx; i++ ) {
				glTexCoord3f( 1.0f*i/texture_sx, 0,                  0 );                  glVertex3i( i,  0, 0 );
				glTexCoord3f( 1.0f*i/texture_sx, 1.0f*sy/texture_sy, 0 );                  glVertex3i( i, sy, 0 );
				glTexCoord3f( 1.0f*i/texture_sx, 1.0f*sy/texture_sy, 1.0f*sz/texture_sz ); glVertex3i( i, sy, sz );
				glTexCoord3f( 1.0f*i/texture_sx, 0,                  1.0f*sz/texture_sz ); glVertex3i( i,  0, sz );
			}
			glEnd();

			glBindTexture( GL_TEXTURE_3D, NULL );
		}

		unsigned int size_x() const { return sx; }
		unsigned int size_y() const { return sy; }
		unsigned int size_z() const { return sz; }
	}; 


	template<class T, class U=uchar>
	class CenterLine : public GLViewer::Object { };

	template<>
	class CenterLine<Edge> : public GLViewer::Object {
		MST::Graph3D<Edge>* ptrTree;
		// Original Data
		int sx;
		int sy;
		int sz;
	public:
		CenterLine( MST::Graph3D<Edge>& tree ) : ptrTree( &tree ) { 
			sx = 0;
			sy = 0;
			sz = 0;
		}

		void init( void ) { }

		void render(void) {
			if( ptrTree==NULL ){
				std::cout << "Failed: Draw Min Span Tree: The Graph is not initialized. " << std::endl; 
				return;
			} 
			if( !ptrTree->num_edges() ) return;
			glBegin( GL_LINES );
			Edge* e = &ptrTree->get_edges().top();
			glColor3f( 1.0f, 1.0f, 1.0f );
			for( int unsigned i=0; i<ptrTree->num_edges(); i++ ) {
				Vec3i p1 = ptrTree->get_pos( e->node1 );
				Vec3i p2 = ptrTree->get_pos( e->node2 );
				glVertex3i( p1[0], p1[1], p1[2] );
				glVertex3i( p2[0], p2[1], p2[2] );
				e++;
			}
			glEnd();
			glColor3f( 0.9f, 0.1f, 0.1f );
		}
		unsigned int size_x() const { return sx; }
		unsigned int size_y() const { return sy; }
		unsigned int size_z() const { return sz; }
	};


	static Graph< MST::Edge_Ext, MST::LineSegment >* ptrTree = NULL;
	static void draw_min_span_tree_init( Graph< MST::Edge_Ext, MST::LineSegment >& tree ) {
		ptrTree = &tree;
	}
	static void draw_min_span_tree(void) {
		if( ptrTree==NULL ){
			std::cout << "Failed: Draw Min Span Tree: The Graph is not initialized. " << std::endl; 
			return;
		} 
		glBegin( GL_LINES );
		MST::Edge_Ext* e = &ptrTree->get_edges().top();
		glColor3f( 1.0f, 0.4f, 0.4f );
		for( int unsigned i=0; i<ptrTree->num_edges(); i++ ) {
			glVertex3f( e->line.p1.x, e->line.p1.y, e->line.p1.z );
			glVertex3f( e->line.p2.x, e->line.p2.y, e->line.p2.z );
			e++;
		}
		glColor3f( 0.4f, 0.4f, 1.0f );
		for( unsigned int i=0; i< ptrTree->num_nodes(); i++ ) {
			MST::LineSegment& line = ptrTree->get_node( i );
			glVertex3f( line.p1.x, line.p1.y, line.p1.z );
			glVertex3f( line.p2.x, line.p2.y, line.p2.z );
		}
		glEnd();
		// glColor3f( 0.5f, 0.5f, 0.5f );
		glColor3f( 0.0f, 1.0f, 0.0f );
	}


	static MST::Graph3D<Edge>* ptrTree2 = NULL;
	static void draw_min_span_tree_init2( MST::Graph3D<Edge>& tree ) {
		ptrTree2 = &tree;
	}
	static void draw_min_span_tree2(void) {
		if( ptrTree2==NULL ){
			std::cout << "Failed: Draw Min Span Tree: The Graph is not initialized. " << std::endl; 
			return;
		} 
		if( !ptrTree2->num_edges() ) return;
		glBegin( GL_LINES );
		Edge* e = &ptrTree2->get_edges().top();
		glColor3f( 1.0f, 1.0f, 1.0f );
		for( int unsigned i=0; i<ptrTree2->num_edges(); i++ ) {
			Vec3i p1 = ptrTree2->get_pos( e->node1 );
			Vec3i p2 = ptrTree2->get_pos( e->node2 );
			glVertex3i( p1[0], p1[1], p1[2] );
			glVertex3i( p2[0], p2[1], p2[2] );
			e++;
		}
		glEnd();
		glColor3f( 0.9f, 0.1f, 0.1f );
	}

	

	///////////////////////////////////////////////////
	// Saving OpenGL Frame Buffer to Video 
	///////////////////////////////////////////////////
	namespace sv{
		string video_name = "temp";
		double duration = 10;
		int current_frame = 0;
		bool isInit = false;
	}
	void save_video_int( string video_name, double frame_rate, double duration ) {
		sv::video_name = video_name;
		sv::duration = duration;
		sv::isInit = true;
	}
	static void save_video( int width, int height ) {
		using namespace sv;

		if( !isInit ) return;

		static int const fps = 20;
		int frames_count = int( fps * sv::duration ); 
		static cv::VideoWriter* outputVideo = NULL;
		if( current_frame==0 ) {
			cout << "Saving Video Begin: Please Stand Still and Don't Touch me!!!" << endl;
			cout << " - File Name: " << sv::video_name << endl;
			cout << " - Frame Rate: " << fps << endl;
			cout << " - Total Number of Frames: " << frames_count << endl;
			cout << " - Frame Size: " << width << " by " << height << endl;

			outputVideo = new cv::VideoWriter( sv::video_name, 
				-1/*CV_FOURCC('M','S','V','C')*/, /*Yuchen: I don't understand this. */
				fps,                    /*Yuchen: Frame Rate */
				cv::Size( width, height ),  /*Yuchen: Frame Size of the Video  */
				true);                      /*Yuchen: Is Color                 */
			if (!outputVideo->isOpened())
			{
				cout  << "Could not open the output video for write: " << endl;
				isInit = false;
				return;
			}
		} else if ( current_frame < frames_count ) {
			Mat pixels( width, height, CV_8UC3 );
			glReadPixels(0, 0, width, height, GL_RGB, GL_UNSIGNED_BYTE, pixels.data );
			Mat cv_pixels( width, height, CV_8UC3 );
			for( int y=0; y<height; y++ ) for( int x=0; x<width; x++ ) 
			{
				cv_pixels.at<Vec3b>(y,x)[2] = pixels.at<Vec3b>(height-y-1,x)[0];
				cv_pixels.at<Vec3b>(y,x)[1] = pixels.at<Vec3b>(height-y-1,x)[1];
				cv_pixels.at<Vec3b>(y,x)[0] = pixels.at<Vec3b>(height-y-1,x)[2];
			}
			(*outputVideo) << cv_pixels; 
		} else if ( current_frame == frames_count ) {
			delete outputVideo;
			outputVideo = NULL;
			isInit = false;
			cout << '\r' << "All Done. Thank you for waiting. " << endl;
			return;
		} 

		sv::current_frame++;
		cout << '\r' << " - Current Frame: " << current_frame;
	}
};


#pragma once

#include "Graph.h"
#include "MinSpanTree.h"
#include "DataTypes.h"

#include <windows.h>		// Header File For Windows
#include <gl\gl.h>			// Header File For The OpenGL32 Library
#include <gl\glu.h>			// Header File For The GLu32 Library
#include <queue>
#include <iostream>

// Using OpenCV ot Save the Scene to video
#define _CRT_SECURE_NO_DEPRECATE
#include <opencv\cv.h>      

namespace GLViewerExt
{
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
		glColor3f( 0.3f, 0.3f, 0.3f );
	}

	namespace sv{
		string video_name = "temp";
		double fps = 20;
		double duration = 10;
		int frames_count = int( fps * duration );
		int current_frame = 0;
		bool isInit = false;
	}
	void save_video_int( string video_name, double frame_rate, double duration ) {
		sv::video_name = video_name;
		sv::fps = frame_rate;
		sv::duration = duration;
		sv::frames_count = int( sv::fps * sv::duration );
		sv::isInit = true;
	}
	static void save_video( int width, int height ) {
		//Mat pixels( width, height, CV_8UC3 );
		//cv::imshow( "filename", pixels.clone() );
		//glReadPixels(0, 0, width, height, GL_RGB, GL_UNSIGNED_BYTE, pixels.clone().data );
		//sv::outputVideo << pixels;
		//return ;

		using namespace sv;

		if( !isInit ) return;

		static cv::VideoWriter* outputVideo = NULL;
		if( current_frame==0 ) {
			cout << "Saving Video Begin: Please Stand Still and Don't Touch me!!!" << endl;
			cout << " - File Name: " << sv::video_name << endl;
			cout << " - Frame Rate: " << sv::fps << endl;
			cout << " - Total Frames: " << sv::frames_count << endl;
			cout << " - Frame Size: " << width << " by " << height << endl;

			outputVideo = new cv::VideoWriter( sv::video_name, 
				-1/*CV_FOURCC('M','S','V','C')*/, /*Yuchen: I don't understand this. */
				sv::fps,                    /*Yuchen: Frame Rate */
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


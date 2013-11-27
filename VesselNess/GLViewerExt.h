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


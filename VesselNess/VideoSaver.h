#pragma once

#include <windows.h>		// Header File For Windows
#include "GLViewer.h"
#include "gl\glew.h"
#include <gl\gl.h>			// Header File For The OpenGL32 Library
#include <gl\glu.h>			// Header File For The GLu32 Library

// Using OpenCV ot Save the Scene to video
#define _CRT_SECURE_NO_DEPRECATE
#include <opencv\cv.h>      

namespace GLViewer
{
	///////////////////////////////////////////////////
	// Saving OpenGL Frame Buffer to Video 
	///////////////////////////////////////////////////
	class VideoSaver : public Object {
		string video_name;
		double duration;
		int current_frame;
		int total_frames;
		bool isInit;
		static int const fps = 20;
	public:
		VideoSaver( string video_name, double duration ) 
			: isInit( true )
			, video_name( video_name )
			, duration( duration)
			, current_frame( 0 )
		{
			total_frames = int( fps * duration ); 
		}

		unsigned int size_x(void) const { return 0; }
		unsigned int size_y(void) const { return 0; }
		unsigned int size_z(void) const { return 0; }

		void render(void) {
			if( !isInit ) return;

			static cv::VideoWriter* outputVideo = NULL;

			int width = 800;
			int height = 800;

			if( current_frame==0 ) {
				cout << "Saving Video Begin: Please Stand Still and Don't Touch me!!!" << endl;
				cout << " - File Name: " << video_name << endl;
				cout << " - Frame Rate: " << fps << endl;
				cout << " - Total Number of Frames: " << total_frames << endl;
				cout << " - Frame Size: " << width << " by " << height << endl;

				outputVideo = new cv::VideoWriter( video_name, 
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
			} else if ( current_frame < total_frames ) {
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
			} else if ( current_frame == total_frames ) {
				delete outputVideo;
				outputVideo = NULL;
				isInit = false;
				cout << '\r' << "All Done. Thank you for waiting. " << endl;
				return;
			} 

			current_frame++;
			cout << '\r' << " - Current Frame: " << current_frame;
		}
	};
}


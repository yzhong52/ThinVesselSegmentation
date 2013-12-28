#define _CRT_SECURE_NO_DEPRECATE
#include "VideoSaver.h"

// Using OpenCV ot Save the Scene to video
#include <opencv\cv.h>
#include <opencv2\opencv.hpp>

// OpenGL Headers
#include <windows.h> // Header File For Windows
#include <gl\gl.h>   // Header File For The OpenGL32 Library
#include <gl\glu.h>  // Header File For The GLu32 Library
#include "GL\freeglut.h"

#include <iostream>

using namespace cv;
using namespace std;

namespace GLViewer
{
	// hidden variable
	static cv::VideoWriter* outputVideo = NULL;

	///////////////////////////////////////////////////
	// Saving OpenGL Frame Buffer to Video 
	///////////////////////////////////////////////////

	void VideoSaver::init(int w, int h ){
		glRotatef( 1.0f, 0.0f, 0.0f, 1.0f );
		// disable reshape function
		// we are not alowed to reshape the window when rendering
		glutReshapeFunc( NULL );

		isInit = true;
		current_frame = 0;
		width = w;
		height = h;

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
			exit(0);
		}
	}

	void VideoSaver::saveBuffer(void) {
		if( autoRotate ) glRotatef( 1, 0, 0, 1 ); 
		if( !isInit ) return;
		
		Mat pixels( /* num of rows */ height, /* num of cols */ width, CV_8UC3 );
		glReadPixels(0, 0, width, height, GL_RGB, GL_UNSIGNED_BYTE, pixels.data );
		Mat cv_pixels( /* num of rows */ height, /* num of cols */ width, CV_8UC3 );
		for( int y=0; y<height; y++ ) for( int x=0; x<width; x++ ) 
		{
			cv_pixels.at<Vec3b>(y,x)[2] = pixels.at<Vec3b>(height-y-1,x)[0];
			cv_pixels.at<Vec3b>(y,x)[1] = pixels.at<Vec3b>(height-y-1,x)[1];
			cv_pixels.at<Vec3b>(y,x)[0] = pixels.at<Vec3b>(height-y-1,x)[2];
		}
		(*outputVideo) << cv_pixels; 
		
		current_frame++;
		cout << '\r' << " - Current Frame: " << current_frame;

		if ( current_frame == total_frames ) {
			delete outputVideo;
			outputVideo = NULL;
			isInit = false;
			cout << '\r' << "All Done. Thank you for waiting. " << endl;
			exit(0);
		} 
	}

	void VideoSaver::takeScreenShot( int w, int h){
		int width  = (int) pow(2, ceil( log(1.0f*w)/log(2.0f) )); 
		int height = (int) pow(2, ceil( log(1.0f*h)/log(2.0f) )); 

		Mat pixels( /* num of rows */ height, /* num of cols */ width, CV_8UC3 );
		glReadPixels(0, 0, width, height, GL_RGB, GL_UNSIGNED_BYTE, pixels.data );

		Mat cv_pixels( /* num of rows */ h, /* num of cols */ w, CV_8UC3 );
		for( int y=0; y<h; y++ ) for( int x=0; x<w; x++ ) 
		{
			cv_pixels.at<Vec3b>(y,x)[2] = pixels.at<Vec3b>(h-y-1,x)[0];
			cv_pixels.at<Vec3b>(y,x)[1] = pixels.at<Vec3b>(h-y-1,x)[1];
			cv_pixels.at<Vec3b>(y,x)[0] = pixels.at<Vec3b>(h-y-1,x)[2];
		}

		stringstream ss; 
		static int index = 0;
		ss << "output/gl_screen_shot" << ++index << ".jpg"; 
		imwrite( ss.str(), cv_pixels ); 
	}
}


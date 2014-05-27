#define _CRT_SECURE_NO_DEPRECATE
#include "GLVideoSaver.h"

// Using OpenCV ot Save the Scene to video
#include "opencv2/core/core.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/video/video.hpp"
#include "opencv2/highgui/highgui.hpp"

/////////////////////////////////////
// Glew Library
#include <GL/glew.h> // For Texture 3D and Blending_Ext
#include <GL/glut.h>

/////////////////////////////////////
// OpenGL Library
#if _MSC_VER && !__INTEL_COMPILER
    #include <windows.h>		// Header File For Windows
    #include <GL\GL.h>			// Header File For The OpenGL32 Library
    #include <GL\glu.h>			// Header File For The GLu32 Library
#endif

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

	void VideoSaver::init(int w, int h, string filename, int maxNumFrames ){
		width = w;
		height = h;
		video_file_name = filename;
		total_frames = maxNumFrames; // total_frames = int( fps * duration );

		// disable reshape function
		// we are not alowed to reshape the window when rendering
		// TODO: should enable reshape function later though
		glutReshapeFunc( NULL );

		state = Rendering;
		current_frame = 0;

		cout << endl << endl;
		cout << "########### GLVideo Saver ########" << endl;
		cout << " Saving Video Begin. Please don't resize the window while video is being generated. " << endl;
		cout << " - File Name: " << video_file_name << endl;
		cout << " - Frame Rate: " << fps << endl;
		cout << " - Total Number of Frames: " << total_frames << endl;
		cout << " - Frame Size: " << width << " by " << height << endl;

		outputVideo = new cv::VideoWriter( video_file_name,
			-1/*CV_FOURCC('M','S','V','C')*/, /*Yuchen: I don't understand this. */
			fps,                    /*Yuchen: Frame Rate */
			cv::Size( width, height ),  /*Yuchen: Frame Size of the Video  */
			true);                      /*Yuchen: Is Color                 */
		if (!outputVideo->isOpened())
		{
			cout  << "Could not open the output video for write: " << endl;
			state = isStopped;
			exit(0);
		}
	}

	void VideoSaver::saveBuffer(void) {
		if( state==isStopped ) return;

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
		cout << '\r' << " - Current Frame: " << current_frame << "\t";

		if ( current_frame==total_frames || state==isAboutToStop ) {
			delete outputVideo; outputVideo = NULL;
			state = isStopped;
			cout << '\r' << "Video '" << video_file_name << "'is saved. ";
			cout << "Thank you for being patient. " << endl;
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


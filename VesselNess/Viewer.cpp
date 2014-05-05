#include "Viewer.h"
#include "Image3D.h"
#include "VesselNessTypes.h"

namespace Viewer{ 
	
	// void plot( Mat_<double> mx, Mat_<double> my );
	namespace OpenCV {
		void plot( const string& name, vector<Mat_<double>>& mat_ys,
			int im_height, int im_width, Mat_<unsigned char> mat_bg )
		{
			// Maximum number of groups supported by the function
			static const int MAX_GROUP = 6;

			// magin of the visualization
			static const float margin = 0.05f;

			int num_ys = (int) mat_ys.size();
			// Error controls
			if( num_ys==0 ) { cout << "Data mat_ys is enmpty" << endl; return;}
			if( num_ys>MAX_GROUP ) { 
				cout << "Cannot handle that many number of groups" << endl; 
				cout << "Maximum number of y supported: " << MAX_GROUP << endl;
				return;
			}

			int width = mat_ys[0].rows;
			// Error Control: Make sure every group has the same width
			for( int i=0; i<num_ys; i++ ){
				if( mat_ys[i].rows!=width ){
					cout << "Error using ==> plot. " << endl << "Vectors must be the same lengths. " << endl; return;
				} 
			}

			if( im_width==0 ) im_width = width;

			double scale = 2.0f * im_width / width;

			// Set the color the lines for each group
			// Maximum number of groups is 6. 
			Scalar colors[MAX_GROUP] = {
				Scalar(0, 0, 255), // red
				Scalar(0, 0, 255), // red
				Scalar(0, 155, 0), // green
				Scalar(255, 0, 0), // blue
				// Scalar(0, 255, 255), // yellow
				Scalar(255, 0, 255), 
				Scalar(255, 255, 0), 
				// Scalar(0, 0, 0)
			};

			// find the maximum and minimum among all mat_y
			Point minLoc, maxLoc;
			// for group 0
			double minVal, maxVal;
			cv::minMaxLoc( mat_ys[0], &minVal, &maxVal );
			// for other groups
			for( int i=1; i<num_ys; i++ ){
				double min_temp, max_temp;
				cv::minMaxLoc( mat_ys[i], &min_temp, &max_temp );
				maxVal = std::max( max_temp, maxVal );
				minVal = std::min( min_temp, minVal );
			}
			double max_min_gap = maxVal - minVal;

			// draw the plot on a mat
			Mat im_bg( int( im_height*scale ), int( width*scale), CV_8UC3, 
				/*Default Background Color*/ Scalar(255, 255, 255) );

			// draw the background
			for( int i=0; i<mat_bg.rows; i++ ){
				unsigned char c = mat_bg.at<unsigned char>(i);
				Scalar color( c, c, c );
				line( im_bg, 
					Point(i, 0)*scale, 
					Point(i, im_height-1)*scale, 
					color, 2, CV_AA, 0 );
			}

			Mat im_result = im_bg.clone();
			for( unsigned int it = 0; it < mat_ys.size(); it++ )
			{
				// Yuchen: draw the image N times for color blending
				// If I draw it only once, the last y data, which is mat_ys[mat_ys.size()]
				// will be drawn on top of all other function. I don't want this kind of bias. 
				Mat temp = im_bg.clone();
				for( unsigned int it2 = 0; it2 < mat_ys.size(); it2++ ) {
					unsigned int i = (it+it2) % mat_ys.size();
					for( int j=1; j < width; j++ ) {
						double v1 = mat_ys[i].at<double>(j-1);
						double v2 = mat_ys[i].at<double>(j);
						Point p1, p2;
						p1.x = int( (j-1) * scale );
						p1.y = int( im_height * scale * ( margin + (1-2*margin)*(1.0 - (v1-minVal)/max_min_gap ) ) );
						p2.x = int( j * scale );
						p2.y = int( im_height * scale * ( margin + (1-2*margin)*(1.0 - (v2-minVal)/max_min_gap ) ) );

						cv::line( temp, p1, p2, colors[i], 3, CV_AA );
					}
				}
				// color blending
				double weight = 1.0 * it / mat_ys.size();
				cv::addWeighted(im_result, weight, temp, 1 - weight, 0, im_result);
			}

			// show result in window and save to file
			imshow( name.c_str(), im_result );
			imwrite( "output/" + name + ".jpg", im_result );
		}
	}
}


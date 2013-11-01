#include "Viewer.h"
#include "Image3D.h"
#include "Vesselness.h"

// Yuchen: Adding glut for OpenCV
#include "gl/glut.h"

// Yuchen: Adding engine for Matlab
#ifdef _CHAR16T
#define CHAR16_T
#endif
#include "engine.h"

namespace Viewer{ 
	namespace OpenGL {

		// global data pointer
		const Data3D<Vesselness_Sig>* ptrVnSig = NULL;
		// Data3D<short>* ptrImage = NULL;
		// setting parameters
		float threshold = 0.0f;
		// Some global variables
		// zooming
		float zoom = 1.0f; 
		// For Camera Rotation
		double camera_angle_h = 0; 
		double camera_angle_v = 0; 
		int drag_x_origin;
		int drag_y_origin;
		int dragging = 0;
		// id for the window that we create
		int window_id;

		//void renderVoxelFunc( const Vec3i& center, const float& thres, int gap = 1){
		//	int x, y, z;
		//	glBegin( GL_POINTS );
		//	glColor3f( 1.0, 1.0, 1.0 );
		//	for( z=0; z < ptrImage->get_size_z(); z+=gap ) {
		//		for( y=0; y < ptrImage->get_size_y(); y+=gap ) {
		//			for( x=0; x < ptrImage->get_size_x(); x+=gap ) {
		//				if( ptrImage->at(x, y, z) > thres ) { 
		//					//// select white color
		//					//glColor3f( 
		//					//	ptrVesselness->at(x,y,z)[0], 
		//					//	ptrVesselness->at(x,y,z)[0], 
		//					//	1.0f - ptrVesselness->at(x,y,z)[0] ); 
		//					// draw the point
		//					glVertex3f( 
		//						zoom*(x-center[0]), 
		//						zoom*(y-center[1]), 
		//						zoom*(z-center[2]));
		//				}
		//			}
		//		}
		//	}
		//	glEnd();
		//}

		void renderDirFunc( void )
		{
			glClear( GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT );

			// calculate the center of the scene
			Vec3i center( ptrVnSig->get_size()/2 );

			// camera
			glLoadIdentity();
			glOrtho( -center[0], center[0], -center[1], center[1], -100000.0f, 100000.0f);
			gluLookAt( /*orgin*/ 0.0, 0.0, 1.0, /*look at*/ 0.0, 0.0, 0.0, /*up vector*/0.0, 1.0, 0.0);

			glRotated( camera_angle_v, 1.0, 0.0, 0.0);
			glRotated( camera_angle_h, 0.0, 1.0, 0.0);

			// draw vessel direction
			glBegin( GL_LINES );
			int gap = 2;
			for( int z=0; z<ptrVnSig->SZ(); z+=gap ) {
				for( int y=0; y<ptrVnSig->SY(); y+=gap ) {
					for( int x=0; x<ptrVnSig->SX(); x+=gap ) {
						if( ptrVnSig->at(x, y, z).rsp > threshold ) { 
							// select line color
							glColor4f( 1.0, 0.0, 0.0, ptrVnSig->at(x,y,z).rsp); 
							// draw line
							glVertex3f( zoom*(x-center[0]), zoom*(y-center[1]), zoom*(z-center[2]) );
							glVertex3f( 
								zoom * ( x - center[0] + ptrVnSig->at(x,y,z)[1] * 5), 
								zoom * ( y - center[1] + ptrVnSig->at(x,y,z)[2] * 5), 
								zoom * ( z - center[2] + ptrVnSig->at(x,y,z)[3] * 5) );
						}
					}
				}
			}
			glEnd();
			glutSwapBuffers();
		}

		void mouse_click(int button, int state, int x, int y) {
			if(button == GLUT_LEFT_BUTTON) {
				if(state == GLUT_DOWN) {
					dragging = 1;
					drag_x_origin = x;
					drag_y_origin = y;
				} else {
					dragging = 0;
				}
			} else if ( button==3 ) {
				// mouse wheel scrolling up
				zoom *= 1.1f;
			} else if ( button==4 ) {
				// mouse wheel scrooling down 
				zoom *= 0.9f;
			}
		}

		void mouse_move(int x, int y) {
			if(dragging) {
				camera_angle_v += (y - drag_y_origin)*0.3;
				camera_angle_h += (x - drag_x_origin)*0.3;
				drag_x_origin = x;
				drag_y_origin = y;
			}
		}

		void reshape( int w, int h )
		{
			glViewport( 0, 0, (GLsizei) w, (GLsizei) h );
			glMatrixMode( GL_PROJECTION );
			glLoadIdentity( );
			glOrtho(0.0, (GLdouble)w, 0.0, (GLdouble)h, -1.0f, 100.0f);
		}

		void key_press( unsigned char key, int x, int y ){
			switch ( key )
			{
			case 27: // Escape key
				glutDestroyWindow ( window_id );
				break;
			}
		}

		void init( void (* renderFunc)( void ) ) 
		{
			int argc = 1;
			char *argv[1] = {(char*)"Something"};
			glutInit(&argc, argv);
			glutInitDisplayMode( GLUT_DOUBLE | GLUT_RGBA );
			glutInitWindowSize( 800, 800 );
			glutInitWindowPosition( 100, 100 );
			window_id = glutCreateWindow( "Visualization of Vesselness" );

			// display
			glutDisplayFunc( renderFunc );
			glutIdleFunc( renderFunc );
			// register mouse fucntions
			glutMouseFunc( mouse_click );
			glutMotionFunc( mouse_move );
			// register keyboard functions
			glutKeyboardFunc( key_press );
			// window resize 
			glutReshapeFunc( reshape );

			glutMainLoop();
		}

		void show_dir( const Data3D<Vesselness_Sig>& vnSig, const float& thres ){
			ptrVnSig = &vnSig;
			threshold = thres;
			smart_return( ptrVnSig->get_size_total(), "Data does not exist." );
			init( renderDirFunc );
		}
	}

	namespace Matlab {
		void plot( Mat_<double> mx, Mat_<double> my )
		{ 
			vector< Mat_<double> > mys( &my, &my+1 );
			plot( mx, mys);
		}

		void plot( Mat_<double> mx, vector< Mat_<double> > mys ){
			for( unsigned int i=0; i<mys.size(); i++ )
			{ 
				if( mx.rows != mys[i].rows ) {
					cout << "Error using ==> plot. " << endl;
					cout << "Vectors must be the same lengths. " << endl;
					return;
				} 
			}
			const int N = mx.rows;

			// open matlab engine
			Engine *ep = engOpen(NULL);
			if ( !ep )
			{
				cout << "Can't start Matlab engine!" <<endl;
				exit(1);
			}

			// The fundamental type underlying MATLAB data
			mxArray *xx = mxCreateDoubleMatrix(1, N, mxREAL); 
			mxArray *yy = mxCreateDoubleMatrix(1, N, mxREAL); 

			// copy data from c++ to matlab data
			memcpy( mxGetPr(xx), mx.data, N * sizeof(double) ); 
			memcpy( mxGetPr(yy), mys[0].data, N * sizeof(double) ); 

			// draw the plot
			engPutVariable( ep, "xx", xx ); 
			engPutVariable( ep, "yy", yy ); 
			
			engEvalString(ep, "figure(1);");
			engEvalString(ep, "hFig = figure(1);;");
			engEvalString(ep, "set(gcf,'PaperPositionMode','auto');");
			engEvalString(ep, "set(hFig, 'Position', [100 100 900 300]);");
			engEvalString(ep, "hFig = plot(xx, yy);");
			engEvalString(ep, "set(gca, 'XTickLabel', num2str(get(gca,'XTick')','%d'));");
			engEvalString(ep, "set(gca, 'YTickLabel', num2str(get(gca,'YTick')','%d'));");

			// release data
			mxDestroyArray(xx); 
			mxDestroyArray(yy); 

			system("pause");

			// close matlab window
			engClose(ep);
		}

		void surf( Mat_<double> matz ){

			// open matlab engine
			Engine *ep = engOpen(NULL);
			if ( !ep )
			{
				cout << "Can't start Matlab engine!" <<endl;
				exit(1);
			}

			// The fundamental type underlying MATLAB data
			mxArray *z = mxCreateDoubleMatrix(matz.cols, matz.rows, mxREAL); 
			
			// copy data from c++ to matlab data
			memcpy( mxGetPr(z), matz.data, matz.rows * matz.cols * sizeof(double) ); 

			// draw the plot
			engPutVariable( ep, "z", z ); 
			
			engEvalString(ep, "axis('axis equal');" );
			engEvalString(ep, "mesh(z);");
			engEvalString(ep, "rotate3d;" );
			engEvalString(ep, "axis('axis equal');" );
			//engEvalString(ep, "figure(1);");
			//engEvalString(ep, "hFig = figure(1);;");
			//engEvalString(ep, "set(gcf,'PaperPositionMode','auto');");
			//engEvalString(ep, "set(hFig, 'Position', [100 100 900 300]);");
			//engEvalString(ep, "hFig = plot(xx, yy);");
			//engEvalString(ep, "set(gca, 'XTickLabel', num2str(get(gca,'XTick')','%d'));");
			//engEvalString(ep, "set(gca, 'YTickLabel', num2str(get(gca,'YTick')','%d'));");

			// release data
			mxDestroyArray(z); 

			system("pause");

			// close matlab window
			engClose(ep);
		}
	}

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

			double scale = 1.0f * im_width / width;

			// Set the color the lines for each group
			// Maximum number of groups is 6. 
			Scalar colors[MAX_GROUP] = {
				Scalar(0, 0, 255), // red
				Scalar(0, 155, 0), // green
				Scalar(255, 0, 0), // blue
				// Scalar(0, 255, 255), // yellow
				Scalar(255, 0, 255), 
				Scalar(255, 255, 0), 
				Scalar(0, 0, 0)
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
					color, 1, CV_AA, 0 );
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

						cv::line( temp, p1, p2, colors[i], 1, CV_AA );
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


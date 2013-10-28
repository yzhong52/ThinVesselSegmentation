#pragma once

#define _CRT_SECURE_NO_DEPRECATE

#include "stdafx.h"

#include "VesselDetector.h"
#include "ImageProcessing.h"





template<typename T> class Data3D;
class Vesselness;
class Vesselness_All;
class Vesselness_Sig;
class Vesselness_Nor;

namespace Viewer
{
	namespace OpenGL{
		void show_dir( const Data3D<Vesselness_Sig>& vnSig, const float& thres = 0.3f );
	}

	namespace MIP{

		template<typename T>
		void Multi_Channels( const Data3D<T>& src, const string& name ) {
			smart_return( src.get_size_total() != 0, "Multi_Channels:: Data cannot be empty."  );
			Image3D<float> im_float;
			src.copyDimTo( im_float, 0 );
			Single_Channel( im_float, name );
		}

		template<typename T>
		void Single_Channel( const Data3D<T>& src, const string& name ) {
			smart_return( src.get_size_total() != 0, "Single_Channel::Data cannot be empty."  );

			Image3D<float> im_float;
			src.convertTo( im_float );
			IP::normalize( im_float, 255.0f );

			Data3D<unsigned char> im_uchar;
			im_float.convertTo( im_uchar );
			im_uchar.save( "temp.data" );

			// save the data as .ive format
			stringstream ss;
			ss << "osgVolume --mip --raw ";
			ss << im_float.get_size_x() << " ";
			ss << im_float.get_size_y() << " ";
			ss << im_float.get_size_z() << " ";
			ss << "1 1 small "; 
			ss << "temp.data ";
			ss << " -o " << name << ".ive" << endl;
			system( ss.str().c_str() );

			// generate bat file
			stringstream ss2;
			ss2 << "osgviewerGlut " << name << ".ive";
			ofstream fout( name+".bat" );
			fout << ss2.str();
			fout.close(); 

			// visualize the data using MIP
			system( ss2.str().c_str() );
		}

	}

	namespace Matlab {
		void plot( Mat_<double> mx, vector< Mat_<double> > mys );
		void plot( Mat_<double> mx, Mat_<double> my );
		void surf( Mat_<double> matz );
	}

	namespace OpenCV {
		void plot( Mat_<double> mx, Mat_<double> my );
		void plot( const string& name,    // name of the output image
			vector<Mat_<double>>& mat_ys, // data y
			int im_height = 400,          // image height
			int im_width = 0,             // image width, will be computed based on the size of mat_ys if left empty
			Mat_<unsigned char> mat_bg = Mat_<unsigned char>());
	}
};

namespace Visualizer = Viewer;
namespace VI = Viewer;


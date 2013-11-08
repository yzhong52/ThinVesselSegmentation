#pragma once

#include "stdafx.h"
#include "Data3D.h"
#include <fstream>


// Image Data (3D, short int)
template<typename T = short>
class Image3D : public Data3D<T>
{
public:
	// constructors and destructors
	Image3D( const Vec3i& n_size=Vec3i(0,0,0) ) : Data3D(n_size), is_roi_set(false)  { }
	virtual ~Image3D(void){}
private:
	Image3D( const Image3D<T>& src ) { smart_assert(0, "Not Implemented."); }

public:
	// read/write data from file
	bool loadData( const string& file_name, const Vec3i& size, const bool& isBigEndian = true, bool isLoadPartial = false );
	inline bool saveData( const string& file_name, bool isBigEndian = true );
	//2D display
	inline void showSlice(int i, const string& name = "Image Data" ); 
	// shink image
	void shrink_by_half(void);
	
public: 
	///////////////////////////////////////////////////////////////
	/////////////////////Region of Interest////////////////////////
	// read/write roi from file
	bool loadROI( const string& roi_file_name );
	bool saveROI( const string& roi_file_name, bool isBigEndian = false );
	// set roi through mouse click
	void setROI(void);
	void setROI( const Vec3i& roi_corner_0, const Vec3i& roi_corner_1 );

	// if roi
	inline Data3D<T>& getROI(void) { return is_roi_set ? roi_data:*this; }
	inline const Data3D<T>& getROI(void) const { return is_roi_set ? roi_data:(*this); }
	// display
	inline void showROI() { roi_data.show(); }
	inline const bool& isROI() const { return is_roi_set; }

	// save as vedio
	bool saveVideo( const string& file_name, Vec<T,2> min_max = Vec<T, 2>(0,0) ) const;

	// get a slice of data (with normalization)
	inline Mat getByZ( const int& z, const T& min, const T& max ) const {
	//	Mat mat_temp = _mat.row(z).reshape( 0, get_height() ).clone(); 
	//	// change the data type from whatever dataype it is to float for computation
	//	mat_temp.convertTo(mat_temp, CV_32F);
	//	// normalize the data range from whatever it is to [0, 255];
	//	mat_temp = 255.0f * ( mat_temp - min ) / (max - min);
	//	// convert back to CV_8U
	//	mat_temp.convertTo(mat_temp, CV_8U);
	//	return mat_temp;
		smart_assert( 0, "deprecated" );
	}

	// get one slice of data
	inline Mat getByZ( const int& z ) const { 
		return getMat().row(z).reshape(0, get_height()).clone(); 
	}

	inline const Vec3i& get_roi_corner_0() const { return roi_corner[0]; }
	inline const Vec3i& get_roi_corner_1() const { return roi_corner[1]; }

private:
	// indicate whether roi is set or not
	bool is_roi_set;
	// Region of Interest (defined by two conner points)
	Vec3i roi_corner[2];
	// roi_data is empty if unset
	Data3D<T> roi_data;
	// Before calling the following function, make sure that the roi_corner[2] are 
	// set properly. And then call the function as
	//		set_roi_from_image( roi_corner[0], roi_corner[1], is_roi_set );
	bool set_roi_from_image(const Vec3i& corner1, const Vec3i& coner2, bool& is_roi_set );


};


template<typename T>
bool Image3D<T>::loadData( const string& file_name, 
	const Vec3i& size, 
	const bool& isBigEndian, bool isLoadPartial )
{
	cout << "loading data from " << file_name << "..." << endl;
	bool flag = load( file_name, size, isBigEndian, isLoadPartial );
	if( !flag ){ 
		cout << "Load fail." << endl << endl;
		return false;
	}
	cout << "done." << endl << endl;

	return true;
}

template<typename T>
bool Image3D<T>::saveData( const string& file_name, bool isBigEndian ){ 
	cout << "Saving data..." << endl;
	smart_return_value( this->_size_total, "Image data is not set. ", false);
	bool flag = save(file_name, isBigEndian);
	cout << "done. " << endl << endl;
	return flag;
}

template<typename T>
bool Image3D<T>::saveVideo( const string& file_name, Vec<T,2> min_max ) const {
	cout << "Saving data as video... " << endl;

	if( min_max == Vec<T,2>(0,0) ){
		min_max = get_min_max_value();
		cout << "Normailizing data from " << min_max << " to " << Vec2i(0, 255) << endl;
	}

	VideoWriter outputVideo;
	outputVideo.open( file_name, -1, 20.0, 
		cv::Size( get_size(0), get_size(1) ), true);
	if (!outputVideo.isOpened())
    {
        cout  << "Could not open the output video for write: " << file_name << endl;
        return false;
    }

	for( int i=0; i<get_size_z(); i++ ){
		Mat im = getByZ(i, min_max[0], min_max[1]);
		outputVideo << im;
	}
	cout << "done." << endl << endl;
	return true;
}


inline void setROI_mouseEvent(int evt, int x, int y, int flags, void* param){
	// get parameters
	int& z = **(int**)param;
	bool& is_roi_init = **((bool**)param + 1);
	Vec3i* roi_corner = *((Vec3i**)param + 2);

	if( evt==CV_EVENT_LBUTTONDOWN ) {
		// If not ROI is not set
		if( !is_roi_init ) {
			roi_corner[0] = roi_corner[1] = Vec3i(x, y, z);
			is_roi_init = true;
		} 
		// Else updata ROI
		else {
			roi_corner[0][0] = min(roi_corner[0][0], x);
			roi_corner[0][1] = min(roi_corner[0][1], y);
			roi_corner[0][2] = min(roi_corner[0][2], z);
			roi_corner[1][0] = max(roi_corner[1][0], x);
			roi_corner[1][1] = max(roi_corner[1][1], y);
			roi_corner[1][2] = max(roi_corner[1][2], z);
		}
	} 
	// If Right Button and ROI is initialized
	else if ( evt==CV_EVENT_RBUTTONDOWN && is_roi_init ) {
		Vec3i roi_center = (roi_corner[0] + roi_corner[1]) /2;
		
		if( x<roi_center[0] ) {
			roi_corner[0][0] = max(roi_corner[0][0], x+1);	
		} else {
			roi_corner[1][0] = min(roi_corner[1][0], x-1);
		}

		if( y<roi_center[1] ) {
			roi_corner[0][1] = max(roi_corner[0][1], y+1);
		} else {
			roi_corner[1][1] = min(roi_corner[1][1], y-1);
		}

		if( z<roi_center[2] ) { 
			roi_corner[0][2] = max(roi_corner[0][2], z+1);
		} else { 
			roi_corner[1][2] = min(roi_corner[1][2], z-1);
		}
	}

	if( (evt==CV_EVENT_LBUTTONDOWN) || (evt==CV_EVENT_RBUTTONDOWN && is_roi_init) ) {
		cout << '\r' << "ROI is from " << roi_corner[0] << " to " << roi_corner[1] << ". ";
		cout << "Size: " << roi_corner[1]-roi_corner[0] << "\t";
	}
}


template<typename T>
void Image3D<T>::setROI( const Vec3i& roi_corner_0, const Vec3i& roi_corner_1 ){
	is_roi_set = true;
	roi_corner[0] = roi_corner_0;
	roi_corner[1] = roi_corner_1;
	set_roi_from_image( roi_corner[0], roi_corner[1], is_roi_set );
}

template<typename T>
void Image3D<T>::setROI(void){
	smart_return( this->get_size_total(), "Image data is not set. ");

	is_roi_set = false;

	static const string window_name = "Setting Region of Interest...";
	
	static const string instructions = string() + \
		"Instructions: \n" + \
		" Left Button (mouse) - include a point \n" + \
		" Right Button (mouse) - exclude a point \n" + \
		" n - next slice \n" + \
		" p - previous slice \n" +\
		" Enter - done \n" +\
		" Exs - reset ";
	
	cout << "Setting Region of Interest" << endl;
	cout << instructions << endl;

	//assigning the callback function for mouse events
	bool is_roi_init = false; // true if ROI is initialized
	int current_slice = 0;
	roi_corner[0] = roi_corner[1] = Vec3i(0, 0, 0);
	void* param[3] = { &current_slice, &is_roi_init, roi_corner };
	namedWindow( window_name.c_str(), CV_WINDOW_AUTOSIZE );
	cvSetMouseCallback( window_name.c_str(), setROI_mouseEvent, param );

	// We are tring to normalize the image data here so that it will be easier
	// for the user to see. 
	// find the maximum and minimum value (method3)
	Point minLoc, maxLoc;
	minMaxLoc( _mat, NULL, NULL, &minLoc, &maxLoc);
	T max_value = _mat.at<T>( maxLoc );
	T min_value = _mat.at<T>( minLoc );

	// displaying data
	do {	
		Mat mat_temp = _mat.row(current_slice).reshape( 0, get_height() ); 
		// change the data type from short to int for computation
		mat_temp.convertTo(mat_temp, CV_32S);
		mat_temp = 255 * ( mat_temp - min_value ) / (max_value - min_value);
		mat_temp.convertTo(mat_temp, CV_8U);
		// Change the data type from GRAY to RGB because we want to dray a 
		// yellow ROI on the data. 
		cvtColor( mat_temp, mat_temp, CV_GRAY2BGR);
		
		if( current_slice>=roi_corner[0][2] && current_slice<=roi_corner[1][2] ) {
			static int i, j;
			for( j=roi_corner[0][1]; j<=roi_corner[1][1]; j++ ) {
				for( i=roi_corner[0][0]; i<=roi_corner[1][0]; i++ )
				{
					// add a yellow shading to ROI
					static const Vec3b yellow(0, 25, 25);
					mat_temp.at<Vec3b>(j, i) += yellow;
				}
			}
		}
		imshow( window_name.c_str(), mat_temp );

		// key controls
		int key = cvWaitKey(250);
		if( key == -1 ) {
			continue;
		} else if( key == 27 ) {
			is_roi_init = false;
			roi_corner[0] = roi_corner[1] = Vec3i(0, 0, 0);
			cout << '\r' << "Region of Interset is reset. \t\t\t\t\t" << endl;
		} else if( key == '\r' ) { 
			break;
		} else if( key == 'n' ){
			if( current_slice < get_depth()-1 ){
				current_slice++;
				cout << '\r' << "Displaying Slice #" << current_slice << "\t\t\t\t\t\t\t";
			}
		} else if( key == 'p') {
			if( current_slice>0 ){
				current_slice--;
				cout << '\r'  << "Displaying Slice #" << current_slice << "\t\t\t\t\t\t\t";
			}
		} else {
			cout << '\r' << "Unknow Input '"<< char(key) << "'. Please follow the instructions below." << endl;
			cout << instructions << endl;
		}
	} while( cvGetWindowHandle(window_name.c_str()) );

	destroyWindow( window_name );

	// set roi data
	if( is_roi_init==false ) {
		cout << "ROI is not set" << endl;
	} else { 
		set_roi_from_image( roi_corner[0], roi_corner[1], is_roi_set );
		cout << '\r' << "ROI is from " << roi_corner[0] << " to " << roi_corner[1] << ". ";
		cout << "Size: " << roi_corner[1]-roi_corner[0] << endl;
	}

	cout << "done. " << endl << endl;
}


template<typename T>
bool Image3D<T>::saveROI( const string& roi_file_name, bool isBigEndian ){
	cout << "Saving ROI..." << endl;
	if( !is_roi_set ){
		cout << "Saving ROI failed. ROI is not set." << endl;
		return false;
	}
	bool flag = roi_data.save( roi_file_name, isBigEndian );
	if( !flag ) return false;
	
	string roi_info_file = roi_file_name + ".readme.txt";
	ofstream fout( roi_info_file.c_str() );
	// dimension of data
	for(int i=0; i<3; i++ ) fout << roi_data.get_size(i) << " ";
	fout << " - size of data" << endl;
	// relative position of the roi with repect to the original data
	for(int i=0; i<3; i++ ) fout << roi_corner[0][i] << " ";
	for(int i=0; i<3; i++ ) fout << roi_corner[1][i] << " ";
	fout << "- position of roi with the respect to the original data" << endl;
	// is the data bigendian: 1 for yes, 0 for no
	fout << isBigEndian << " - Big Endian (1 for yes, 0 for no)" << endl;
	// data type
	if( typeid(short)==typeid(T) ) fout << "short";
	fout << " - data type" << endl;
	fout.close();

	cout << "done. " << endl << endl;
	return true;
}


template<typename T>
bool Image3D<T>::loadROI( const string& roi_file_name )
{
	string roi_info_file = roi_file_name + ".readme.txt";
	ifstream fin( roi_info_file.c_str() );

	cout << "Loading ROI info from '"<< roi_info_file << "'..." << endl;

	if( !fin.is_open() ) {
		cout << "The ROI Info file not found. " << endl;
		return false;
	}

	// load roi info
	// size of data
	Vec3i roi_size;
	for(int i=0; i<3; i++ ) fin >> roi_size[i];
	fin.ignore(256,'\n');   // ignore the rest of the line
	// relative position of the roi with repect to the original data
	for(int i=0; i<3; i++ ) fin >> roi_corner[0][i];
	for(int i=0; i<3; i++ ) fin >> roi_corner[1][i];
	fin.ignore(256,'\n');   // ignore the rest of the line
	// is the data bigendian: 1 for yes, 0 for no
	bool isBigEndian;
	fin >> isBigEndian;
	fin.ignore(256,'\n');   // ignore the rest of the line
	// data type
	string datatype;
	fin >> datatype;
	smart_return_value( !datatype.compare( STR_TYPE(typeid(T)) ), "datatype not matched.", false );
	fin.close();

	// set roi_data
	// if data is loaded, copy roi data from image data
	if( this->get_size_total() ) {
		cout << "Setting ROI from image data..." << endl; 
		set_roi_from_image( roi_corner[0], roi_corner[1], is_roi_set );
	} 
	// else, load roi from data file
	else {
		cout << "Image data is not set, loading ROI from file '" << roi_file_name << "'..." << endl;
		is_roi_set = roi_data.load( roi_file_name, roi_size, isBigEndian );	
	}
	cout << "done." << endl << endl;
	return is_roi_set;
	
}

template<typename T>
void Image3D<T>::showSlice(int i, const string& name = "Image Data" ){
	Mat mat_temp = _mat.row(i).reshape( 0, get_height() ); 
	cv::imshow( name.c_str(), mat_temp );
	waitKey(0);
}

template<typename T>
void Image3D<T>::shrink_by_half(void){	
	smart_return( this->size_total, "Image data is not set. ");

	Vec3i n_size = size / 2;
	int n_size_slice = n_size[0] * n_size[1];
	int n_size_total = n_size_slice * n_size[2];

	// We need to add two short number, which may result in overflow. 
	// Therefore, we use CV_64S for safety
	Mat n_mat = Mat( n_size[2], n_size_slice, mat.type(), Scalar(0) );
	int i, j, k;
	for( i=0; i<n_size[0]; i++ ) for( j=0; j<n_size[1]; j++ ) for( k=0; k<n_size[2]; k++ )
	{
		n_mat.at<T>(k, j*n_size[0]+i)  = 0.25 * mat.at<T>(2*k,     2*j*size[0] + 2*i);
		n_mat.at<T>(k, j*n_size[0]+i) += 0.25 * mat.at<T>(2*k,     2*j*size[0] + 2*i + 1);
		n_mat.at<T>(k, j*n_size[0]+i) += 0.25 * mat.at<T>(2*k + 1, 2*j*size[0] + 2*i);
		n_mat.at<T>(k, j*n_size[0]+i) += 0.25 * mat.at<T>(2*k + 1, 2*j*size[0] + 2*i + 1);
	}
	mat = n_mat;

	size = n_size;
	size_slice = n_size_slice;
	size_total = n_size_total;
}

template<typename T>
bool Image3D<T>::set_roi_from_image(const Vec3i& corner1, const Vec3i& corner2, bool& is_roi_set ){
	roi_data.reset( roi_corner[1]-roi_corner[0]+Vec3i(1,1,1) );
	if( roi_data.get_size_total()==0 ) {
		cout << "Region of Interest is not initialzied properly." << endl;
		is_roi_set = false;
	} else { 
		// copy image data to roi_data
		Vec3i roi_pos;
		for( roi_pos[2]=0; roi_pos[2]<roi_data.get_size_z(); roi_pos[2]++ ) {
			for( roi_pos[1]=0; roi_pos[1]<roi_data.get_size_y(); roi_pos[1]++ ) {
				for( roi_pos[0]=0; roi_pos[0]<roi_data.get_size_x(); roi_pos[0]++ ) {
					roi_data.at( roi_pos ) = this->at( roi_corner[0] + roi_pos );
				}
			}
		}
		is_roi_set = true;
	}
	return is_roi_set;
}

///////////////////////////////////////////////////////////////////////////
// Global Functions

template<typename T>
void remove_margin(const Data3D<T>& src, Data3D<T>& dst, Vec3i margin = Vec3i(10,10,10) ){
	dst.reset( src.get_size()-2*margin );
	int x, y, z;
	for( z=0; z<dst.get_size_z(); z++ ){
		for( y=0; y<dst.get_size_y(); y++ ){
			for( x=0; x<dst.get_size_x(); x++ ){
				dst.at(x,y,z) = src.at(x+margin[0], y+margin[1], z+margin[2]);
			}
		}
	}
}


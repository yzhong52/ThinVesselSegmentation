#pragma once

#include "stdafx.h"

template<typename T>
class Data3D {
public:
	// Constructors & Destructors
	Data3D(){ }
	Data3D( const Vec3i& n_size ){ 
		reset(n_size); 
	}	
	Data3D( const Vec3i& n_size, const T& value ){ 
		reset(n_size, value);
	}	
	// this copy constructor is extremely similar to the copyTo function
	template <class T2>
	Data3D( const Data3D<T2>& src ) {
		this->resize( src.get_size() );

		cv::Mat_<T>::iterator        this_it;
		cv::Mat_<T2>::const_iterator src_it;
		for( this_it = this->getMat().begin(), src_it = src.getMat().begin(); 
			 this_it < this->getMat().end(),   src_it < src.getMat().end();
			 this_it++, src_it++ )
		{
			*(this_it) = *(src_it);
		}
	}
	virtual ~Data3D(void){ }


	//////////////////////////////////////////////////////////////////////
	// reset the data
	virtual void reset( const Vec3i& n_size, const T& value )
	{
		resize( n_size );
		for( Mat_<T>::iterator it=_mat.begin(); it<_mat.end(); it++ ) {
			(*it) = value;
		}
	}
	
	virtual void reset( const Vec3i& n_size ){
		_size = n_size;
		_size_slice = _size[0] * _size[1];
		_size_total = _size_slice * _size[2];
		_mat = Mat_<T>( _size[2], _size_slice );
		memset( _mat.data, 0, _size_total * sizeof(T) ); 
	}
	virtual void reset( void ) {
		memset( _mat.data, 0, _size_total * sizeof(T) );
	}

	virtual void resize( const Vec3i& n_size )
	{
		_size = n_size;
		_size_slice = _size[0] * _size[1];
		_size_total = _size_slice * _size[2];
		_mat = Mat_<T>( _size[2], _size_slice );
	}

	
	// getters about the size of the data
	inline const int& SX(void) const { return _size[0]; }
	inline const int& SY(void) const { return _size[1]; }
	inline const int& SZ(void) const { return _size[2]; }
	inline const int& get_width(void)  const { return _size[0]; }
	inline const int& get_height(void) const { return _size[1]; }
	inline const int& get_depth(void)  const { return _size[2]; }
	inline const int& get_size_x(void) const { return _size[0]; }
	inline const int& get_size_y(void) const { return _size[1]; }
	inline const int& get_size_z(void) const { return _size[2]; }
	inline const int& get_size(int i) const { return _size[i]; }
	inline const int& get_size_slice(void) const { return _size_slice; }
	inline const int& get_size_total(void) const { return _size_total; }
	inline const Vec3i& get_size(void) const { return _size; }
	
	// getters about values of the data
	virtual const inline T& at( const int& x, const int& y, const int& z ) const { 
		return _mat.at<T>( z, y * _size[0] + x ); 
	}
	virtual inline T& at( const int& x, const int& y, const int& z ) { 
		return _mat.at<T>( z, y * _size[0] + x ); 
	}
	virtual const inline T& at( const Vec3i& pos ) const {
		return at( pos[0], pos[1], pos[2] );
	}
	virtual inline T& at( const Vec3i& pos ) {
		return at( pos[0], pos[1], pos[2] );
	}

	// getter of the data
	inline const Mat_<T>& getMat() const { return _mat; }
	inline Mat_<T>& getMat() { return _mat; }
	
	// loading/saving data from/to file
	bool load( const string& file_name );
	bool load( const string& file_name, const Vec3i& size, bool isBigEndian=true, bool isLoadPartial=false );
	bool save( const string& file_name, bool isBigEndian = false, bool saveInfo = true ) const;
	void show( const string& window_name = "Show 3D Data by Slice", int current_slice = 0 ) const;

	// overwrite operators
	template<typename T1, typename T2>
	friend const Data3D<T1>& operator*=( Data3D<T1>& left, const T2& right );
	template<typename T1, typename T2, typename T3>
	friend bool subtract3D( const Data3D<T1>& src1, const Data3D<T2>& src2, Data3D<T3>& dst );
	template<typename T1, typename T2, typename T3>
	friend bool multiply3D( const Data3D<T1>& src1, const Data3D<T2>& src2, Data3D<T3>& dst );
	
	
	// change the type of the data
	template< typename DT >
	void convertTo( Data3D<DT>& dst ) const {
		dst.reset( this->get_size() );
		this->getMat().convertTo( dst.getMat(), CV_TYPE( typeid(DT) ) );
	}

	// get minimum and maximum value of the data
	Vec<T, 2> get_min_max_value() const {
		Vec<T, 2> min_max;
		Point minLoc, maxLoc;
		minMaxLoc( _mat, NULL, NULL, &minLoc, &maxLoc);
		min_max[0] = _mat.at<T>( minLoc );
		min_max[1] = _mat.at<T>( maxLoc );
		return min_max;
	}

	public:
	// copy data of dimension i to a new Image3D structure
	// e.g. this is useful when trying to visualize vesselness which is
	// a multidimensional data structure
	// Yuchen: p.s. I have no idea how to move the funciton body out the class
	// definition nicely. Let me figure it later maybe. 
	template<typename T2>
	void copyDimTo( Data3D<T2>& dst, int dim ) const {
		dst.reset( this->get_size() );
		int x, y, z;
		for( z=0; z<get_size_z(); z++ ) for( y=0; y<get_size_y(); y++ ) for( x=0; x<get_size_x(); x++ ) {
			dst.at(x,y,z) = this->at(x,y,z)[dim];
		}
	}
	inline bool remove_margin( const int& margin ) {
		return remove_margin( Vec3i( margin, margin, margin) );
	}
	bool remove_margin( const Vec3i& margin ) {
		Vec3i n_size;
		for( int i=0; i<3; i++ ){
			n_size[i] = _size[i] - 2 * margin[i];
			if( n_size[i] <= 0 ) {
				cout << "Margin is too big. Remove margin failed. " << endl;
				return false;
			}
		}
		int n_size_slice = n_size[0] * n_size[1];
		int n_size_total = n_size[2] * n_size_slice;
		Mat_<T> n_mat = Mat_<T>( n_size[2], n_size_slice );

		for( int z=margin[2]; z<_size[2]-margin[2]; z++ ){
			for( int y=margin[1]; y<_size[1]-margin[1]; y++ ){
				for( int x=margin[0]; x<_size[0]-margin[0]; x++ ){
					n_mat.at<T>( z-margin[2], (y-margin[1])*n_size[0] + (x-margin[0]) ) = _mat.at<T>( z, y*_size[0] + x );
				}
			}
		}
		_mat = n_mat.clone();
		_size = n_size;
		_size_slice = n_size_slice;
		_size_total = n_size_total;
		return true;
	}

	void remove_margin( const Vec3i& margin1, const Vec3i& margin2 ) {
		Vec3i n_size;
		for( int i=0; i<3; i++ ){
			n_size[i] = _size[i] - margin1[i] - margin2[i];
			if( n_size[i] <= 0 ) {
				cout << "Margin is too big. Remove margin failed. " << endl;
				return;
			}
		}
		int n_size_slice = n_size[0] * n_size[1];
		int n_size_total = n_size[2] * n_size_slice;
		Mat_<T> n_mat = Mat_<T>( n_size[2], n_size_slice );

		for( int z=margin1[2]; z<_size[2]-margin2[2]; z++ ){
			for( int y=margin1[1]; y<_size[1]-margin2[1]; y++ ){
				for( int x=margin1[0]; x<_size[0]-margin2[0]; x++ ){
					n_mat.at<T>( z-margin1[2], (y-margin1[1])*n_size[0] + (x-margin1[0]) ) = _mat.at<T>( z, y*_size[0] + x );
				}
			}
		}
		_mat = n_mat.clone();
		_size = n_size;
		_size_slice = n_size_slice;
		_size_total = n_size_total;
	}

	// return true if a index is valide for the data
	bool isValid( const int& x, const int& y, const int& z ) const {
		return ( x>=0 && x<get_size_x() &&
			     y>=0 && y<get_size_y() &&
			     z>=0 && z<get_size_z() );
	}
	bool isValid( const Vec3i& v ) const { 
		return isValid( v[0], v[1], v[2] );
	}

protected:
	// Maximum size of the x, y and z direction respetively. 
	Vec3i _size;
	// int size_x, size_y, size_z;
	// size_x * size_y * size_z
	int _size_total;   
	// size_x * size_y
	int _size_slice; 
	// data <T>
	Mat_<T> _mat;

private:
	// TODO: I will try yxml later
	void save_info( const string& file_name, bool isBigEndian ) const;
	bool load_info( const string& file_name, Vec3i& size, bool& isBigEndian );
};

template <typename T>
bool Data3D<T>::save( const string& file_name, bool isBigEndian, bool saveInfo ) const
{
	smart_return_value( _size_total, "Data is empty", false );

	FILE* pFile=fopen( file_name.c_str(), "wb" );
	if( isBigEndian ) {
		smart_return_value( sizeof(T)==2, "Datatype is not supported.", false );
		Mat temp_mat = _mat.clone();
		// swap the data
		unsigned char* temp = temp_mat.data;
		for(int i=0; i<_size_total; i++) {
			std::swap( *temp, *(temp+1) );
			temp+=2;
		}
		fwrite( temp_mat.data, sizeof(T), _size_total, pFile );
	} else {
		fwrite( _mat.data, sizeof(T), _size_total, pFile );
	}
	fclose(pFile);

	if( saveInfo ) save_info( file_name, isBigEndian );
	return true;
}


template <typename T>
bool Data3D<T>::load( const string& file_name ){
	// load info
	bool isBigEndian;
	Vec3i size;
	bool flag = load_info( file_name, size, isBigEndian );
	if( !flag ) return false;
	// load data
	return load( file_name, size, isBigEndian );
}

template <typename T>
bool Data3D<T>::load( const string& file_name, const Vec3i& size, bool isBigEndian, bool isLoadPartial )
{
	// reset size of the data
	reset( size );
	
	// loading data from file
	FILE* pFile=fopen( file_name.c_str(), "rb" );
	smart_return_value( pFile!=0, "File not found", false );

	int size_read = (int) fread( _mat.data, sizeof(T), _size_total, pFile);
	
	fgetc( pFile );
	// if we haven't read the end of the file
	// and if we specify that we only want to load part of the data (isLoadPartial)
	if( !feof(pFile) && !isLoadPartial ) {
		fclose(pFile);
		cout << "Data size is incorrect (too small)" << endl;
		return false;
	}
	fclose(pFile);
	
	// if the size we read is not as big as the size we expected, fail
	smart_return_value( size_read==_size_total, "Data size is incorrect (too big)", false );
	
	if( isBigEndian ) {
		smart_return_value( sizeof(T)==2, "Datatype is not supported.", false );
		// swap the data
		unsigned char* temp = _mat.data;
		for(int i=0; i<_size_total; i++) {
			std::swap( *temp, *(temp+1) );
			temp+=2;
		}
	}

	return true;
}



template<typename T>
void Data3D<T>::show(const string& window_name, int current_slice ) const 
{
	smart_return( this->get_size_total(), "Data is empty." );
	smart_return( 
		typeid(T)==typeid(float)||
		typeid(T)==typeid(short)||
		typeid(T)==typeid(int)||
		typeid(T)==typeid(unsigned char)||
		typeid(T)==typeid(unsigned short), 
		"Datatype cannot be visualized." );

	namedWindow( window_name.c_str(), CV_WINDOW_AUTOSIZE );

	static const string instructions = string() + \
		"Instructions: \n" + \
		" n - next slice \n" + \
		" p - previous slice \n" +\
		" s - save the current slice \n" +\
		" Exc - exit ";
	
	cout << "Displaying data by slice. " << endl;
	cout << instructions << endl;

	// find the maximum and minimum value (method3)
	Point minLoc, maxLoc;
	minMaxLoc( _mat, NULL, NULL, &minLoc, &maxLoc);
	T max_value = _mat.at<T>( maxLoc );
	T min_value = _mat.at<T>( minLoc );

	cout << "Displaying Slice #" << current_slice;
	do {
		Mat mat_temp = _mat.row(current_slice).reshape( 0, get_height() ).clone(); 
		// change the data type from whatever dataype it is to float for computation
		mat_temp.convertTo(mat_temp, CV_32F);
		// normalize the data range from whatever it is to [0, 255];
		mat_temp = 255.0f * ( mat_temp - min_value ) / (max_value - min_value);
		// convert back to CV_8U for visualizaiton
		mat_temp.convertTo(mat_temp, CV_8U);
		// show in window
		cv::imshow( window_name.c_str(), mat_temp );
		// cv::imshow( window_name.c_str(), _mat.row(current_slice).reshape( 0, get_height() ) );
		// key controls
		int key = cvWaitKey(0);
		// Yuchen: Program will stop at cvWaitKey above and User may
		// close the windows at this moment. 
		if( !cvGetWindowHandle( window_name.c_str()) ) break;
		if( key == 27 ) {
			break;
		} else if( key == 'n' ){
			if( current_slice < get_depth()-1 ){
				current_slice++;
				cout << '\r' << "Displaying Slice #" << current_slice << "\t\t\t\t\t\t";
			}
		} else if( key == 'p') {
			if( current_slice>0 ){
				current_slice--;
				cout << '\r' << "Displaying Slice #" << current_slice << "\t\t\t\t\t\t";
			}
		} else if ( key =='s' ) {
			stringstream ss;
			ss << "Original_Data_Slice_"; 
			ss.width(3); ss.fill('0');
			ss << current_slice << ".jpg"; 
			cv::imwrite( ss.str(), mat_temp );
		} else {
			cout << '\r' << "Unknow Input '"<< char(key) << "'. Please follow the instructions above.";
		}
	} while( cvGetWindowHandle( window_name.c_str()) );
	destroyWindow( window_name.c_str() );
	cout << endl << "done." << endl << endl;
}


template<typename T>
void Data3D<T>::save_info( const string& file_name, bool isBigEndian ) const {
	string info_file = file_name + ".readme.txt";
	ofstream fout( info_file.c_str() );
	fout << _size[0] << " ";
	fout << _size[1] << " ";
	fout << _size[2] << " - data size" << endl;
	fout << STR_TYPE( typeid(T) ) << " - data type" << endl;
	fout << isBigEndian << " - Big Endian (1 for yes, 0 for no)" << endl;
	fout.close();
}


template<typename T>
bool Data3D<T>::load_info( const string& file_name, Vec3i& size, bool& isBigEndian ) {
	string info_file = file_name + ".readme.txt";
	ifstream fin( info_file.c_str() );
	// size
	if( !fin.is_open() ) return false;
	fin >> size[0];
	fin >> size[1]; 
	fin >> size[2]; 
	fin.ignore(255, '\n');
	// type 
	string str_type;
	fin >> str_type;
	fin.ignore(255, '\n');
	if( STR_TYPE(typeid(T)).compare( str_type ) != 0 ){
		cout << "Loading information error: Data types unmatch. File: " << info_file << endl;
		return false;
	}
	// endian
	fin >> isBigEndian;
	fin.ignore(255, '\n');
	// close the file
	fin.close();
	return true;
}


//////////////////////////////////////////////////////////////////////
// Operator Overloading and other friend functions

template<typename T, typename T2>
const Data3D<T>& operator*=( Data3D<T>& left, const T2& right ){
	left._mat*=right;
	return left;
}

template<typename T1, typename T2, typename T3>
bool subtract3D( const Data3D<T1>& src1, const Data3D<T2>& src2, Data3D<T3>& dst ){
	smart_return_false( src1.get_size()==src2.get_size(),
		"Source sizes are supposed to be matched.");

	if( dst.get_size() != src1.get_size() ){
		dst.reset( src1.get_size() );
	}

	int x, y, z;
	for( z=0; z<src1.get_size_z(); z++ ) {
		for( y=0; y<src1.get_size_y(); y++ ) {
			for( x=0; x<src1.get_size_x(); x++ ) {
				dst.at(x,y,z) = src1.at(x,y,z) - src2.at(x,y,z);
			}
		}
	}
	return true;
}


template<typename T1, typename T2, typename T3>
bool multiply3D( const Data3D<T1>& src1, const Data3D<T2>& src2, Data3D<T3>& dst ){
	smart_return_false( src1.get_size()==src2.get_size(),
		"Source sizes are supposed to be matched.");

	if( dst.get_size() != src1.get_size() ){
		dst.reset( src1.get_size() );
	}

	int x, y, z;
	for( z=0; z<src1.get_size_z(); z++ ) {
		for( y=0; y<src1.get_size_y(); y++ ) {
			for( x=0; x<src1.get_size_x(); x++ ) {
				dst.at(x,y,z) = src1.at(x,y,z) * src2.at(x,y,z);
			}
		}
	}
	return true;
}


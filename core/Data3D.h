#pragma once

//#include "stdafx.h"
#include "TypeInfo.h"
#include "nstdio.h"

#include <iostream>
#include <fstream> // For reading and saving files
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <typeinfo>
#include "smart_assert.h"

template<typename T>
class Data3D
{
public:
    //////////////////////////////////////////////////////////////////////
    // Constructors & Destructors
    // Default Constructor
    Data3D( const cv::Vec3i& n_size = cv::Vec3i(0,0,0));
    // Constructor with size and value
    Data3D( const cv::Vec3i& n_size, const T& value );
    // Constructor from file
    Data3D( const std::string& filename );
    // Copy Constructor - extremely similar to the copyTo function
    template <class T2>
    Data3D( const Data3D<T2>& src );
    // Destructor
    virtual ~Data3D(void) { }

    //////////////////////////////////////////////////////////////////////
    // reset the data
    void reset( const cv::Vec3i& n_size, const T& value )
    {
        resize( n_size );
        for( cv::MatIterator_<T> it=_mat.begin(); it<_mat.end(); it++ )
        {
            (*it) = value;
        }
    }

    inline void reset( const cv::Vec3i& n_size )
    {
        _size = n_size;
        _size_slice = _size[0] * _size[1];
        _size_total = _size_slice * _size[2];
        _mat = cv::Mat_<T>( _size[2], _size_slice );
        memset( _mat.data, 0, _size_total * sizeof(T) );
    }
    inline void reset( void )
    {
        memset( _mat.data, 0, _size_total * sizeof(T) );
    }

    void resize( const cv::Vec3i& n_size )
    {
        _size = n_size;
        _size_slice = _size[0] * _size[1];
        _size_total = _size_slice * _size[2];
        _mat = cv::Mat_<T>( _size[2], _size_slice );
    }

    // getters about the size of the data
    inline const int& SX(void) const
    {
        return _size[0];
    }
    inline const int& SY(void) const
    {
        return _size[1];
    }
    inline const int& SZ(void) const
    {
        return _size[2];
    }
    inline const int& get_width(void)  const
    {
        return _size[0];
    }
    inline const int& get_height(void) const
    {
        return _size[1];
    }
    inline const int& get_depth(void)  const
    {
        return _size[2];
    }
    inline const int& get_size_x(void) const
    {
        return _size[0];
    }
    inline const int& get_size_y(void) const
    {
        return _size[1];
    }
    inline const int& get_size_z(void) const
    {
        return _size[2];
    }
    inline const int& get_size(int i) const
    {
        return _size[i];
    }
    inline const long& get_size_slice(void) const
    {
        return _size_slice;
    }
    inline const long& get_size_total(void) const
    {
        return _size_total;
    }
    inline const cv::Vec3i& get_size(void) const
    {
        return _size;
    }

    // getters about values of the data
    virtual const inline T& at( const int& i ) const
    {
        return _mat( i );
    }
    virtual inline T& at( const int& i )
    {
        return _mat( i );
    }
    virtual inline const T& at( const int& x, const int& y, const int& z ) const
    {
        return _mat( z, y * _size[0] + x );
    }
    virtual inline T& at( const int& x, const int& y, const int& z )
    {
        return _mat( z, y * _size[0] + x );
    }
    virtual inline const T& at( const cv::Vec3i& pos ) const
    {
        return at( pos[0], pos[1], pos[2] );
    }
    virtual inline T& at( const cv::Vec3i& pos )
    {
        return at( pos[0], pos[1], pos[2] );
    }

    // getter of the data
    inline cv::Mat_<T>& getMat()
    {
        return _mat;
    }
    inline const cv::Mat_<T>& getMat() const
    {
        return _mat;
    }
    inline cv::Mat_<T> getMat( int slice ) const
    {
        return _mat.row(slice).reshape( 0, get_height() ).clone();
    }

    // loading/saving data from/to file
    bool load( const std::string& file_name );
    bool load( const std::string& file_name, const cv::Vec3i& size,
               bool isBigEndian=true, bool isLoadPartial=false );
    bool save( const std::string& file_name, const std::string& log = "",
               bool saveInfo = true, bool isBigEndian = false ) const;
    void show( const std::string& window_name = "Show 3D Data by Slice",
               int current_slice = 0 ) const;
    void show( const std::string& window_name, int current_slice,
               T max_value, T min_value ) const;

    // overwrite operators
    template<typename T1, typename T2>
    friend const Data3D<T1>& operator*=( Data3D<T1>& left, const T2& right );
    template<typename T1, typename T2, typename T3>
    friend bool subtract3D( const Data3D<T1>& src1, const Data3D<T2>& src2, Data3D<T3>& dst );
    template<typename T1, typename T2, typename T3>
    friend bool multiply3D( const Data3D<T1>& src1, const Data3D<T2>& src2, Data3D<T3>& dst );

    // change the type of the data
    template< typename DT >
    void convertTo( Data3D<DT>& dst ) const
    {
        dst.reset( this->get_size() );
        this->getMat().convertTo( dst.getMat(), TypeInfo<DT>::CV_TYPE() );
    }

    // get minimum and maximum value of the data
    cv::Vec<T, 2> get_min_max_value() const
    {
        cv::Vec<T, 2> min_max;
        cv::Point minLoc, maxLoc;
        cv::minMaxLoc( _mat, NULL, NULL, &minLoc, &maxLoc);
        min_max[0] = _mat( minLoc );
        min_max[1] = _mat( maxLoc );
        return min_max;
    }

public:
    // copy data of dimension i to a new Image3D structure
    // e.g. this is useful when trying to visualize vesselness which is
    // a multidimensional data structure
    // Yuchen: p.s. I have no idea how to move the funciton body out the class
    // definition nicely. Let me figure it later maybe.
    template<typename T2>
    void copyDimTo( Data3D<T2>& dst, int dim ) const
    {
        dst.reset( this->get_size() );
        int x, y, z;
        for( z=0; z<get_size_z(); z++ ) for( y=0; y<get_size_y(); y++ ) for( x=0; x<get_size_x(); x++ )
                {
                    dst.at(x,y,z) = this->at(x,y,z)[dim];
                }
    }

    // remove some margin of the data
    inline bool remove_margin( const int& margin );
    // remove some margin of the data
    // such that left_margin and right_margin are the same
    inline bool remove_margin( const cv::Vec3i& margin );
    // remove some margin of the data
    bool remove_margin( const cv::Vec3i& margin1, const cv::Vec3i& margin2 );
    // Resize the data by cropping or padding
    // This is done through remove margin func
    void remove_margin_to( const cv::Vec3i& size );


    // return true if a index is valide for the data
    inline bool isValid( const int& x, const int& y, const int& z ) const
    {
        return ( x>=0 && x<get_size_x() &&
                 y>=0 && y<get_size_y() &&
                 z>=0 && z<get_size_z() );
    }
    inline bool isValid( const cv::Vec3i& v ) const
    {
        return isValid( v[0], v[1], v[2] );
    }

    inline const T* getData(void) const
    {
        return (T*) getMat().data;
    }

    inline T* getData(void)
    {
        return (T*) getMat().data;
    }

protected:
    // Maximum size of the x, y and z direction respetively.
    cv::Vec3i _size;
    // int size_x, size_y, size_z;
    // size_x * size_y * size_z
    long _size_total;
    // size_x * size_y
    long _size_slice;
    // data <T>
    cv::Mat_<T> _mat;

private:
    // TODO: I should try yxml later
    void save_info( const std::string& file_name, bool isBigEndian, const std::string& log )  const;
    bool load_info( const std::string& file_name, cv::Vec3i& size, bool& isBigEndian );
};


// Default Constructor
template <typename T>
Data3D<T>::Data3D( const cv::Vec3i& n_size )
{
    reset(n_size);
}

// Constructor with size and value
template <typename T>
Data3D<T>::Data3D( const cv::Vec3i& n_size, const T& value )
{
    reset(n_size, value);
}

// Constructor with a given file
template<typename T>
Data3D<T>::Data3D( const std::string& filename )
{
    bool flag = load( filename );
    if( !flag )
    {
        std::cerr << "Data3D<T>::Data3D::Cannot load the file. " << std::endl;
    }
}

// Copy Constructor - extremely similar to the copyTo function
template<typename T>
template<typename T2>
Data3D<T>::Data3D( const Data3D<T2>& src )
{
    // resize
    this->resize( src.get_size() );
    // copy the data over
    cv::MatIterator_<T>       this_it;
    cv::MatConstIterator_<T2> src_it;
    for( this_it = this->getMat().begin(), src_it = src.getMat().begin();
            this_it < this->getMat().end(),   src_it < src.getMat().end();
            this_it++, src_it++ )
    {
        // Yuchen: The convertion from T2 to T may be unsafe
        *(this_it) = T( *(src_it) );
    }
}

template <typename T>
bool Data3D<T>::save( const std::string& file_name, const std::string& log, bool saveInfo, bool isBigEndian ) const
{
    smart_return( _size_total, "Save File Failed: Data is empty", false );

    std::cout << "Saving file to " << file_name << std::endl;
    std::cout << "Data Size: " << (long) _size_total * sizeof(T) << " bytes "<< std::endl;

    FILE* pFile = fopen( file_name.c_str(), "wb" );
    if( isBigEndian )
    {
        if( sizeof(T)!=2 )
        {
            std::cout << "Save File Failed: Datatype does not supported Big Endian. ";
            std::cout << "Please contact Yuchen for more detail. " << std::endl;
            return false;
        }
        cv::Mat temp_mat = _mat.clone();
        // swap the data
        unsigned char* temp = temp_mat.data;
        for(int i=0; i<_size_total; i++)
        {
            std::swap( *temp, *(temp+1) );
            temp+=2;
        }
        fwrite_big( temp_mat.data, sizeof(T), _size_total, pFile );
    }
    else
    {
        fwrite_big( _mat.data, sizeof(T), _size_total, pFile );
    }
    fclose(pFile);

    // saving the data information to a txt file
    if(saveInfo) save_info( file_name, isBigEndian, log );

    std::cout << "done." << std::endl << std::endl;
    return true;
}


template <typename T>
bool Data3D<T>::load( const std::string& file_name )
{
    // load data information
    bool isBigEndian;
    cv::Vec3i size;
    bool flag = load_info( file_name, size, isBigEndian );
    if( !flag ) exit(0);
    // load data
    return load( file_name, size, isBigEndian );
}

template <typename T>
bool Data3D<T>::load( const std::string& file_name, const cv::Vec3i& size, bool isBigEndian, bool isLoadPartial )
{
    std::cout << "Loading Data '" << file_name << "'" << std::endl;

    // reset size of the data
    reset( size );

    // loading data from file
    FILE* pFile=fopen( file_name.c_str(), "rb" );
    smart_return( pFile!=0, "File not found", false );

    unsigned long long size_read = fread_big( _mat.data, sizeof(T), _size_total, pFile);

    fgetc( pFile );
    // if we haven't read the end of the file
    // and if we specify that we only want to load part of the data (isLoadPartial)
    if( !feof(pFile) && !isLoadPartial )
    {
        fclose(pFile);
        std::cout << "Data size is incorrect (too small)" << std::endl;
        return false;
    }
    fclose(pFile);

    // if the size we read is not as big as the size we expected, fail
    smart_return( size_read==_size_total*sizeof(T),
                  "Data size is incorrect (too big)", false );

    if( isBigEndian )
    {
        smart_return( sizeof(T)==2, "Datatype does not support big endian.", false );
        // swap the data
        unsigned char* temp = _mat.data;
        for(int i=0; i<_size_total; i++)
        {
            std::swap( *temp, *(temp+1) );
            temp+=2;
        }
    }

    std::cout << "Done." << std::endl << std::endl;
    return true;
}


template<typename T>
void Data3D<T>::show(const std::string& window_name, int current_slice, T min_value, T max_value ) const
{
    smart_assert( this->get_size_total(), "Data is empty." );
    smart_assert(
        typeid(T)==typeid(float)||
        typeid(T)==typeid(short)||
        typeid(T)==typeid(int)||
        typeid(T)==typeid(unsigned char)||
        typeid(T)==typeid(unsigned short),
        "Datatype cannot be visualized." );

    cv::namedWindow( window_name.c_str(), CV_WINDOW_AUTOSIZE );

    static const std::string instructions = std::string() + \
                                            "Instructions: \n" + \
                                            " n - next slice \n" + \
                                            " p - previous slice \n" +\
                                            " s - save the current slice \n" +\
                                            " Exc - exit ";

    std::cout << "Displaying data by slice. " << std::endl;
    std::cout << instructions << std::endl;
    std::cout << "Displaying Slice #" << current_slice;

    if( current_slice > SZ() ) current_slice = SZ();
    do
    {
        cv::Mat mat_temp = _mat.row(current_slice).reshape( 0, get_height() ).clone();
        // change the data type from whatever dataype it is to float for computation
        mat_temp.convertTo(mat_temp, CV_32F);
        // normalize the data range from whatever it is to [0, 255];
        mat_temp = 255.0f * (  mat_temp - min_value ) / (max_value - min_value);
        // convert back to CV_8U for visualizaiton
        mat_temp.convertTo(mat_temp, CV_8U);
        // show in window
        cv::imshow( window_name.c_str(),  mat_temp );

        // key controls
        int key = cvWaitKey(0);

        // This following is in order to adjust an undocumented bug in
        // OpenCV. See this post for more details:
        // http://stackoverflow.com/questions/9172170/python-opencv-cv-waitkey-spits-back-weird-output-on-ubuntu-modulo-256-maps-corre
        key = key & 255;

        // Yuchen: Program will stop at cvWaitKey above and User may
        // close the windows at this moment.
        if( !cvGetWindowHandle( window_name.c_str()) ) break;
        if( key == 27 )
        {
            break;
        }
        else if( key == 'n' || key=='N' )
        {
            if( current_slice < get_depth()-1 )
            {
                current_slice++;
                std::cout << '\r' << "Displaying Slice #" << current_slice << "\t\t\t\t\t\t";
                std::cout.flush();
            }
        }
        else if( key=='p' || key=='P' )
        {
            if( current_slice>0 )
            {
                current_slice--;
                std::cout << '\r' << "Displaying Slice #" << current_slice << "\t\t\t\t\t\t";
                std::cout.flush();
            }
        }
        else if ( key=='s' || key=='S' )
        {
            std::stringstream ss;
            ss << "output/" << window_name << "_Data_Slice_";
            ss.width(3);
            ss.fill('0');
            ss << current_slice << ".jpg";
            bool flag = cv::imwrite( ss.str(),  mat_temp );
            if( flag )
            {
                std::cout << "A screen shot is save as: '" << ss.str() << "'" << std::endl;
            }
            else
            {
                std::cerr << "Failed to save a screen shot is save as: '" << ss.str() << "'" << std::endl;
            }
            std::cout.flush();
        }
        else
        {
            std::cout << '\r' << "Unknow Input '"<< char(key) << "'. Please follow the instructions above.";
        }
    }
    while( cvGetWindowHandle( window_name.c_str()) );
    cv::destroyWindow( window_name.c_str() );

    std::cout << std::endl << "done." << std::endl << std::endl;
}

template<typename T>
void Data3D<T>::show(const std::string& window_name, int current_slice ) const
{
    // find the maximum and minimum values
    cv::Point minLoc, maxLoc;
    minMaxLoc( _mat, NULL, NULL, &minLoc, &maxLoc);
    T max_value = _mat( maxLoc );
    T min_value = _mat( minLoc );

    this->show( window_name, current_slice, min_value, max_value );
}


template<typename T>
void Data3D<T>::save_info( const std::string& file_name, bool isBigEndian, const std::string& log  ) const
{
    std::string info_file = file_name + ".readme.txt";
    std::cout << "Saving data information to '" << info_file << "' " << std::endl;
    std::ofstream fout( info_file.c_str() );
    fout << _size[0] << " ";
    fout << _size[1] << " ";
    fout << _size[2] << " - data size" << std::endl;
    fout << TypeInfo<T>::str() << " - data type" << std::endl;
    fout << isBigEndian << " - Big Endian (1 for yes, 0 for no)" << std::endl;
    fout << "Log: " <<  log << std::endl;
    fout.close();
}


template<typename T>
bool Data3D<T>::load_info( const std::string& file_name, cv::Vec3i& size, bool& isBigEndian )
{
    // data info name
    std::string info_file = file_name + ".readme.txt";
    std::cout << "Loading data information from '" << info_file << "' " << std::endl;

    // open file
    std::ifstream fin( info_file.c_str() );
    if( !fin.is_open() )
    {
        std::cout << "The readme file: '" << info_file << "' is not found." << std::endl;
        return false;
    }
    // size of the data
    fin >> size[0];
    fin >> size[1];
    fin >> size[2];
    fin.ignore(255, '\n');
    // data type
    std::string str_type;
    fin >> str_type;
    fin.ignore(255, '\n');
    if( TypeInfo<T>::str().compare( str_type ) != 0 )
    {
        std::cout << "Loading information error: ";
        std::cout << "Data3D<" << TypeInfo<T>::str() << "> cannot load Data3D<" << str_type << ">. "<< std::endl;
        return false;
    }
    // endian of the data
    fin >> isBigEndian;
    fin.ignore(255, '\n');
    // close the file
    fin.close();
    return true;
}


//////////////////////////////////////////////////////////////////////
// Operator Overloading and other friend functions

template<typename T, typename T2>
const Data3D<T>& operator*=( Data3D<T>& left, const T2& right )
{
    left._mat*=right;
    return left;
}

template<typename T1, typename T2, typename T3>
bool subtract3D( const Data3D<T1>& src1, const Data3D<T2>& src2, Data3D<T3>& dst )
{
    smart_return( src1.get_size()==src2.get_size(),
                  "Source sizes are supposed to be cv::Matched.", false);

    if( dst.get_size() != src1.get_size() )
    {
        dst.reset( src1.get_size() );
    }

    int x, y, z;
    for( z=0; z<src1.get_size_z(); z++ )
    {
        for( y=0; y<src1.get_size_y(); y++ )
        {
            for( x=0; x<src1.get_size_x(); x++ )
            {
                dst.at(x,y,z) = src1.at(x,y,z) - src2.at(x,y,z);
            }
        }
    }
    return true;
}


template<typename T1, typename T2, typename T3>
bool multiply3D( const Data3D<T1>& src1, const Data3D<T2>& src2, Data3D<T3>& dst )
{
    smart_return_false( src1.get_size()==src2.get_size(),
                        "Source sizes are supposed to be cv::Matched.");

    if( dst.get_size() != src1.get_size() )
    {
        dst.reset( src1.get_size() );
    }

    int x, y, z;
    for( z=0; z<src1.get_size_z(); z++ )
    {
        for( y=0; y<src1.get_size_y(); y++ )
        {
            for( x=0; x<src1.get_size_x(); x++ )
            {
                dst.at(x,y,z) = src1.at(x,y,z) * src2.at(x,y,z);
            }
        }
    }
    return true;
}



// remove some margin of the data
template<typename T>
inline bool Data3D<T>::remove_margin( const int& margin )
{
    return remove_margin( cv::Vec3i( margin, margin, margin) );
}

// remove some margin of the data
// such that left_margin and right_margin are the same
template<typename T>
inline bool Data3D<T>::remove_margin( const cv::Vec3i& margin )
{
    return remove_margin( margin, margin );
}

// remove some margin of the data
template<typename T>
bool Data3D<T>::remove_margin( const cv::Vec3i& margin1, const cv::Vec3i& margin2 )
{
    cv::Vec3i n_size;
    for( int i=0; i<3; i++ )
    {
        n_size[i] = _size[i] - margin1[i] - margin2[i];
        if( n_size[i] <= 0 )
        {
            std::cout << "Margin is too big. Remove margin failed. " << std::endl;
            return false;
        }
    }

    // allocate memory for new data
    int n_size_slice = n_size[0] * n_size[1];
    int n_size_total = n_size[2] * n_size_slice;
    cv::Mat_<T> n_mat = cv::Mat_<T>( n_size[2], n_size_slice );

    // Remove a negetive margin is equivalent to
    // padding zeros around the data
    cv::Vec3i spos = cv::Vec3i(
                         std::max( margin1[0], 0),
                         std::max( margin1[1], 0),
                         std::max( margin1[2], 0)
                     );
    cv::Vec3i epos = cv::Vec3i(
                         std::min( _size[0]-margin2[0], _size[0] ),
                         std::min( _size[1]-margin2[1], _size[1] ),
                         std::min( _size[2]-margin2[2], _size[2] )
                     );
    for( int z=spos[2]; z<epos[2]; z++ )
    {
        for( int y=spos[1]; y<epos[1]; y++ )
        {
            for( int x=spos[0]; x<epos[0]; x++ )
            {
                n_mat( z-margin1[2], (y-margin1[1])*n_size[0] + (x-margin1[0]) ) = _mat( z, y*_size[0] + x );
            }
        }
    }
    // update the data
    // just passing the reference here is good
    _mat = n_mat;
    // updata the size
    _size = n_size;
    _size_slice = n_size_slice;
    _size_total = n_size_total;
    return true;
}

// Resize the data by cropping or padding
// This is done through remove margin func
template<typename T>
void Data3D<T>::remove_margin_to( const cv::Vec3i& size )
{
    const cv::Vec3i left(
        (int) floor(1.0f*(SX()-size[0])/2),
        (int) floor(1.0f*(SY()-size[1])/2),
        (int) floor(1.0f*(SZ()-size[2])/2) );
    const cv::Vec3i right( (int) ceil(1.0f*(SX()-size[0])/2),  (int) ceil(1.0f*(SY()-size[1])/2), (int) ceil(1.0f*(SZ()-size[2])/2)) ;
    remove_margin( left, right );
}

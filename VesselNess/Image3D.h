#pragma once

#include "Data3D.h"
#include <fstream>
#include <opencv2/imgproc/imgproc.hpp>

// Image Data (3D, short int)
template<typename T = short>
class Image3D : public Data3D<T>
{
public:
    // constructors and destructors
    Image3D( const cv::Vec3i& n_size=cv::Vec3i(0,0,0) ) : Data3D<T>(n_size), is_roi_set(false)  { }
    virtual ~Image3D(void) {}
private:
    Image3D( const Image3D<T>& src )
    {
        /*smart_assert(0, "Not Implemented."); */
    }

public:
    // read/write data from file
    bool loadData( const std::string& file_name, const cv::Vec3i& size, const bool& isBigEndian = true, bool isLoadPartial = false );
    inline bool saveData( const std::string& file_name, bool isBigEndian = true );
    //2D display
    inline void showSlice(int i, const std::string& name = "Image Data" );
    // shink image
    void shrink_by_half(void);

public:
    ///////////////////////////////////////////////////////////////
    /////////////////////Region of Interest////////////////////////
    // read/write roi from file
    bool loadROI( const std::string& roi_file_name );
    bool saveROI( const std::string& roi_file_name, const std::string& log="", bool isBigEndian = false );
    // set roi through mouse click
    void setROI(void);
    void setROI( const cv::Vec3i& roi_corner_0, const cv::Vec3i& roi_corner_1 );

    // if roi
    inline Data3D<T>& getROI(void)
    {
        return is_roi_set ? roi_data:*this;
    }
    inline const Data3D<T>& getROI(void) const
    {
        return is_roi_set ? roi_data:(*this);
    }
    // display
    inline void showROI()
    {
        roi_data.show();
    }
    inline const bool& isROI() const
    {
        return is_roi_set;
    }

    // save as vedio
    bool saveVideo( const std::string& file_name, cv::Vec<T,2> min_max = cv::Vec<T, 2>(0,0) ) const;

    // get a slice of data (with normalization)
    inline cv::Mat getByZ( const int& z, const T& min, const T& max ) const
    {
        cv::Mat mat_temp = this->_mat.row(z).reshape( 0, this->get_height() ).clone();
        // change the data type from whatever dataype it is to float for computation
        mat_temp.convertTo(mat_temp, CV_32F);
        // normalize the data range from whatever it is to [0, 255];
        mat_temp = 255.0f * ( mat_temp - min ) / (max - min);
        // convert back to CV_8U
        mat_temp.convertTo(mat_temp, CV_8U);
        return mat_temp;
        // smart_assert( 0, "deprecated" );
    }

    // get one slice of data
    inline cv::Mat getByZ( const int& z ) const
    {
        return this->getMat().row(z).reshape(0, this->get_height()).clone();
    }

    inline const cv::Vec3i& get_roi_corner_0() const
    {
        return roi_corner[0];
    }
    inline const cv::Vec3i& get_roi_corner_1() const
    {
        return roi_corner[1];
    }

private:
    // indicate whether roi is set or not
    bool is_roi_set;
    // Region of Interest (defined by two conner points)
    cv::Vec3i roi_corner[2];
    // roi_data is empty if unset
    Data3D<T> roi_data;
    // Before calling the following function, make sure that the roi_corner[2] are
    // set properly. And then call the function as
    //		set_roi_from_image( roi_corner[0], roi_corner[1], is_roi_set );
    bool set_roi_from_image(const cv::Vec3i& corner1, const cv::Vec3i& coner2, bool& is_roi_set );
};


template<typename T>
bool Image3D<T>::loadData( const std::string& file_name,
                           const cv::Vec3i& size,
                           const bool& isBigEndian, bool isLoadPartial )
{
    std::cout << "loading data from " << file_name << "..." << std::endl;
    bool flag = this->load( file_name, size, isBigEndian, isLoadPartial );
    if( !flag )
    {
        std::cout << "Load fail." << std::endl << std::endl;
        return false;
    }
    std::cout << "done." << std::endl << std::endl;

    return true;
}

template<typename T>
bool Image3D<T>::saveData( const std::string& file_name, bool isBigEndian )
{
    std::cout << "Saving data..." << std::endl;
    smart_return_value( this->_size_total, "Image data is not set. ", false);
    bool flag = this->save(file_name, isBigEndian);
    std::cout << "done. " << std::endl << std::endl;
    return flag;
}

template<typename T>
bool Image3D<T>::saveVideo( const std::string& file_name, cv::Vec<T,2> min_max ) const
{
    std::cout << "Saving data as video... " << std::endl;

    if( min_max == cv::Vec<T,2>(0,0) )
    {
        min_max = this->get_min_max_value();
        std::cout << "Normailizing data from " << min_max << " to " << cv::Vec2i(0, 255) << std::endl;
    }

    cv::VideoWriter outputVideo;
    outputVideo.open( file_name, -1, 20.0,
                      cv::Size( this->get_size(0), this->get_size(1) ), true);
    if (!outputVideo.isOpened())
    {
        std::cout  << "Could not open the output video for write: " << file_name << std::endl;
        return false;
    }

    for( int i=0; i < this->get_size_z(); i++ )
    {
        cv::Mat im = getByZ(i, min_max[0], min_max[1]);
        outputVideo << im;
    }
    std::cout << "done." << std::endl << std::endl;
    return true;
}


inline void setROI_mouseEvent(int evt, int x, int y, int flags, void* param)
{
    // get parameters
    int& z = **(int**)param;
    bool& is_roi_init = **((bool**)param + 1);
    cv::Vec3i* roi_corner = *((cv::Vec3i**)param + 2);

    if( evt==CV_EVENT_LBUTTONDOWN )
    {
        // If not ROI is not set
        if( !is_roi_init )
        {
            roi_corner[0] = roi_corner[1] = cv::Vec3i(x, y, z);
            is_roi_init = true;
        }
        // Else updata ROI
        else
        {
            roi_corner[0][0] = std::min(roi_corner[0][0], x);
            roi_corner[0][1] = std::min(roi_corner[0][1], y);
            roi_corner[0][2] = std::min(roi_corner[0][2], z);
            roi_corner[1][0] = std::max(roi_corner[1][0], x);
            roi_corner[1][1] = std::max(roi_corner[1][1], y);
            roi_corner[1][2] = std::max(roi_corner[1][2], z);
        }
    }
    // If Right Button and ROI is initialized
    else if ( evt==CV_EVENT_RBUTTONDOWN && is_roi_init )
    {
        cv::Vec3i roi_center = (roi_corner[0] + roi_corner[1]) /2;

        if( x<roi_center[0] )
        {
            roi_corner[0][0] = std::max(roi_corner[0][0], x+1);
        }
        else
        {
            roi_corner[1][0] = std::min(roi_corner[1][0], x-1);
        }

        if( y<roi_center[1] )
        {
            roi_corner[0][1] = std::max(roi_corner[0][1], y+1);
        }
        else
        {
            roi_corner[1][1] = std::min(roi_corner[1][1], y-1);
        }

        if( z<roi_center[2] )
        {
            roi_corner[0][2] = std::max(roi_corner[0][2], z+1);
        }
        else
        {
            roi_corner[1][2] = std::min(roi_corner[1][2], z-1);
        }
    }

    if( (evt==CV_EVENT_LBUTTONDOWN) || (evt==CV_EVENT_RBUTTONDOWN && is_roi_init) )
    {
        std::cout << '\r' << "ROI is from " << roi_corner[0] << " to " << roi_corner[1] << ". ";
        std::cout << "Size: " << roi_corner[1]-roi_corner[0] << "\t";
    }
}


template<typename T>
void Image3D<T>::setROI( const cv::Vec3i& roi_corner_0, const cv::Vec3i& roi_corner_1 )
{
    is_roi_set = true;
    roi_corner[0] = roi_corner_0;
    roi_corner[1] = roi_corner_1;
    set_roi_from_image( roi_corner[0], roi_corner[1], is_roi_set );
}

template<typename T>
void Image3D<T>::setROI(void)
{
    smart_return( this->get_size_total(), "Image data is not set. ");

    is_roi_set = false;

    static const std::string window_name = "Setting Region of Interest...";

    static const std::string instructions = std::string() + \
                                       "Instructions: \n" + \
                                       " Left Button (mouse) - include a point \n" + \
                                       " Right Button (mouse) - exclude a point \n" + \
                                       " n - next slice \n" + \
                                       " p - previous slice \n" +\
                                       " Enter - done \n" +\
                                       " Exs - reset ";

    std::cout << "Setting Region of Interest" << std::endl;
    std::cout << instructions << std::endl;

    //assigning the callback function for mouse events
    bool is_roi_init = false; // true if ROI is initialized
    int current_slice = 0;
    roi_corner[0] = roi_corner[1] = cv::Vec3i(0, 0, 0);
    void* param[3] = { &current_slice, &is_roi_init, roi_corner };
    cv::namedWindow( window_name.c_str(), CV_WINDOW_AUTOSIZE );
    cvSetMouseCallback( window_name.c_str(), setROI_mouseEvent, param );

    // We are tring to normalize the image data here so that it will be easier
    // for the user to see.
    // find the maximum and minimum value (method3)
    cv::Point minLoc, maxLoc;
    minMaxLoc( Data3D<T>::_mat, NULL, NULL, &minLoc, &maxLoc);
    T max_value = this->_mat( maxLoc );
    T min_value = this->_mat( minLoc );

    // displaying data
    do
    {
        cv::Mat mat_temp = Data3D<T>::_mat.row(current_slice).reshape( 0, Data3D<T>::get_height() );
        // change the data type from short to int for computation
        mat_temp.convertTo(mat_temp, CV_32S);
        mat_temp = 255 * ( mat_temp - min_value ) / (max_value - min_value);
        mat_temp.convertTo(mat_temp, CV_8U);
        // Change the data type from GRAY to RGB because we want to dray a
        // yellow ROI on the data.
        cvtColor( mat_temp, mat_temp, CV_GRAY2RGB);

        if( current_slice>=roi_corner[0][2] && current_slice<=roi_corner[1][2] )
        {
            static int i, j;
            for( j=roi_corner[0][1]; j<=roi_corner[1][1]; j++ )
            {
                for( i=roi_corner[0][0]; i<=roi_corner[1][0]; i++ )
                {
                    // add a yellow shading to ROI
                    static const cv::Vec3b yellow(0, 25, 25);
                    mat_temp.at<cv::Vec3b>(j, i) += yellow;
                }
            }
        }
        imshow( window_name.c_str(), mat_temp );

        // key controls
        int key = cvWaitKey(250);
        if( key == -1 )
        {
            continue;
        }
        else if( key == 27 )
        {
            is_roi_init = false;
            roi_corner[0] = roi_corner[1] = cv::Vec3i(0, 0, 0);
            std::cout << '\r' << "Region of Interset is reset. \t\t\t\t\t" << std::endl;
        }
        else if( key == '\r' )
        {
            break;
        }
        else if( key == 'n' )
        {
            if( current_slice < Data3D<T>::get_depth()-1 )
            {
                current_slice++;
                std::cout << '\r' << "Displaying Slice #" << current_slice << "\t\t\t\t\t\t\t";
            }
        }
        else if( key == 'p')
        {
            if( current_slice>0 )
            {
                current_slice--;
                std::cout << '\r'  << "Displaying Slice #" << current_slice << "\t\t\t\t\t\t\t";
            }
        }
        else
        {
            std::cout << '\r' << "Unknow Input '"<< char(key) << "'. Please follow the instructions below." << std::endl;
            std::cout << instructions << std::endl;
        }
    }
    while( cvGetWindowHandle(window_name.c_str()) );

    cv::destroyWindow( window_name );

    // set roi data
    if( is_roi_init==false )
    {
        std::cout << "ROI is not set" << std::endl;
    }
    else
    {
        set_roi_from_image( roi_corner[0], roi_corner[1], is_roi_set );
        std::cout << '\r' << "ROI is from " << roi_corner[0] << " to " << roi_corner[1] << ". ";
        std::cout << "Size: " << roi_corner[1]-roi_corner[0] << std::endl;
    }

    std::cout << "done. " << std::endl << std::endl;
}


template<typename T>
bool Image3D<T>::saveROI( const std::string& roi_file_name, const std::string& log, bool isBigEndian )
{
    std::cout << "Saving ROI..." << std::endl;
    if( !is_roi_set )
    {
        std::cout << "Saving ROI failed. ROI is not set." << std::endl;
        return false;
    }
    bool flag = roi_data.save( roi_file_name, log, /*save info*/false, isBigEndian );
    if( !flag ) return false;

    std::string roi_info_file = roi_file_name + ".readme.txt";
    std::ofstream fout( roi_info_file.c_str() );

    // size of data
    for(int i=0; i<3; i++ ) fout << roi_data.get_size(i) << " ";
    fout << " - size of data" << std::endl;

    // data type
    fout << TypeInfo<T>::str() << " - data type" << std::endl;

    // is the data bigendian: 1 for yes, 0 for no
    fout << isBigEndian << " - Big Endian (1 for yes, 0 for no)" << std::endl;

    // relative position of the roi with repect to the original data
    for(int i=0; i<3; i++ ) fout << roi_corner[0][i] << " ";
    for(int i=0; i<3; i++ ) fout << roi_corner[1][i] << " ";
    fout << " - position of roi with the respect to the original data" << std::endl;

    fout << log << " - log"<< std::endl;

    fout.close();

    std::cout << "done. " << std::endl << std::endl;
    return true;
}


template<typename T>
bool Image3D<T>::loadROI( const std::string& roi_file_name )
{
    std::string roi_info_file = roi_file_name + ".readme.txt";
    std::ifstream fin( roi_info_file.c_str() );

    if( !fin.is_open() )
    {
        std::cout << "The ROI Info file not found. " << std::endl;
        return false;
    }

    // size of data
    cv::Vec3i roi_size;
    for(int i=0; i<3; i++ ) fin >> roi_size[i];
    fin.ignore(256,'\n');   // ignore the rest of the line

    // data type
    std::string datatype;
    fin >> datatype;
    fin.ignore(255, '\n');
    if( TypeInfo<T>::str().compare( datatype ) != 0 )
    {
        std::cout << "Loading information error: ";
        std::cout << "Data3D<" << TypeInfo<T>::str() << "> cannot load Data3D<" << datatype << ">. "<< std::endl;
        return false;
    }

    // is the data bigendian: 1 for yes, 0 for no
    bool isBigEndian;
    fin >> isBigEndian;
    fin.ignore(256,'\n');   // ignore the rest of the line

    // relative position of the roi with repect to the original data
    for(int i=0; i<3; i++ ) fin >> roi_corner[0][i];
    for(int i=0; i<3; i++ ) fin >> roi_corner[1][i];
    fin.ignore(256,'\n');   // ignore the rest of the line

    // close file
    fin.close();

    // set roi_data
    // if data is loaded, copy roi data from image data
    if( this->get_size_total() )
    {
        std::cout << "Setting ROI from image data..." << std::endl;
        set_roi_from_image( roi_corner[0], roi_corner[1], is_roi_set );
    }
    // else, load roi from data file
    else
    {
        std::cout << "Image data is not set, loading ROI from file '" << roi_file_name << "'..." << std::endl;
        is_roi_set = roi_data.load( roi_file_name, roi_size, isBigEndian );
    }
    std::cout << "done." << std::endl << std::endl;
    return is_roi_set;

}

template<typename T>
void Image3D<T>::showSlice(int i, const std::string& name )
{
    cv::Mat mat_temp = Data3D<T>::_mat.row(i).reshape( 0, Data3D<T>::get_height() );
    cv::imshow( name.c_str(), mat_temp );
    cv::waitKey(0);
}

template<typename T>
void Image3D<T>::shrink_by_half(void)
{
    smart_return( this->_size_total, "Image data is not set. ");

    cv::Vec3i n_size = (Data3D<T>::_size - cv::Vec3i(1,1,1)) / 2; // TODO: why do I have to minute cv::Vec3i(1,1,1) here?
    int n_size_slice = n_size[0] * n_size[1];
    int n_size_total = n_size_slice * n_size[2];

    // We need to add two short number, which may result in overflow.
    // Therefore, we use CV_64S for safety
    cv::Mat n_mat = cv::Mat( n_size[2], n_size_slice, Data3D<T>::_mat.type(), cv::Scalar(0) );
    int i, j, k;
    for( i=0; i<n_size[0]; i++ ) for( j=0; j<n_size[1]; j++ ) for( k=0; k<n_size[2]; k++ )
            {
                n_mat.at<T>(k, j*n_size[0]+i)  = T( 0.25 * Data3D<T>::_mat(2*k,     2 * j * Data3D<T>::_size[0] + 2 * i) );
                n_mat.at<T>(k, j*n_size[0]+i) += T( 0.25 * Data3D<T>::_mat(2*k,     2 * j * Data3D<T>::_size[0] + 2 * i + 1) );
                n_mat.at<T>(k, j*n_size[0]+i) += T( 0.25 * Data3D<T>::_mat(2*k + 1, 2 * j * Data3D<T>::_size[0] + 2 * i) );
                n_mat.at<T>(k, j*n_size[0]+i) += T( 0.25 * Data3D<T>::_mat(2*k + 1, 2 * j * Data3D<T>::_size[0] + 2 * i + 1) );
            }
    Data3D<T>::_mat = n_mat;

    Data3D<T>::_size = n_size;
    Data3D<T>::_size_slice = n_size_slice;
    Data3D<T>::_size_total = n_size_total;
}

template<typename T>
bool Image3D<T>::set_roi_from_image(const cv::Vec3i& corner1, const cv::Vec3i& corner2, bool& is_roi_set )
{
    roi_data.reset( roi_corner[1]-roi_corner[0]+cv::Vec3i(1,1,1) );
    if( roi_data.get_size_total()==0 )
    {
        std::cout << "Region of Interest is not initialzied properly." << std::endl;
        is_roi_set = false;
    }
    else
    {
        // copy image data to roi_data
        cv::Vec3i roi_pos;
        for( roi_pos[2]=0; roi_pos[2]<roi_data.get_size_z(); roi_pos[2]++ )
        {
            for( roi_pos[1]=0; roi_pos[1]<roi_data.get_size_y(); roi_pos[1]++ )
            {
                for( roi_pos[0]=0; roi_pos[0]<roi_data.get_size_x(); roi_pos[0]++ )
                {
                    roi_data.at( roi_pos ) = this->at( roi_corner[0] + roi_pos );
                }
            }
        }
        is_roi_set = true;
    }
    return is_roi_set;
}


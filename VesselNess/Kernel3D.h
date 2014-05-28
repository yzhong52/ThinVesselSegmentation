#pragma once
#include <opencv2/imgproc/imgproc.hpp>

template<typename T> class Data3D;

template<typename T = float>
class Kernel3D : public Data3D<T>
{
    // operator overload
    template <typename U>
    friend std::ostream& operator<<(std::ostream& out, const Kernel3D<U>& data);

public:
    // constructor and destructors
    Kernel3D() {};
    Kernel3D( const cv::Vec3i& n_size );
    virtual ~Kernel3D() {}
    // getters
    const int& min_pos(int i) const
    {
        return min_p[i];
    }
    const int& max_pos(int i) const
    {
        return max_p[i];
    }
    virtual const inline T& offset_at(const int& x, const int& y, const int& z) const
    {
        return Data3D<T>::at(x+center[0], y+center[1], z+center[2]);
    }
    virtual inline T& offset_at(const int& x, const int& y, const int& z)
    {
        return Data3D<T>::at(x+center[0], y+center[1], z+center[2]);
    }

    // setters
    void reset( const cv::Vec3i& n_size, cv::Scalar scalar = cv::Scalar(0));
private:
    // minimun and maximum possition
    cv::Vec3i min_p, max_p, center;
public:
    // Some Global Functions for Getting 3D Kernel's that are commonly used.
    static Kernel3D<T> GaussianFilter3D( cv::Vec3i size );
    inline static Kernel3D<T> GaussianFilter3D( int ksize )
    {
        smart_assert( typeid(T)==typeid(float) || typeid(T)==typeid(double),
                      "Can only use float or double" );
        return GaussianFilter3D( cv::Vec3i(ksize, ksize, ksize) );
    }
    inline static Kernel3D<T> dx()
    {
        smart_assert( typeid(T)==typeid(float) || typeid(T)==typeid(double),
                      "Can only use float or double" );
        Kernel3D<T> dx( cv::Vec3i(3,1,1) );
        dx.at( 0, 0, 0 ) = -0.5;
        dx.at( 2, 0, 0 ) =  0.5;
        return dx;
    }
    inline static Kernel3D<T> dy()
    {
        smart_assert( typeid(T)==typeid(float) || typeid(T)==typeid(double),
                      "Can only use float or double" );
        Kernel3D<T> dy( cv::Vec3i(1,3,1) );
        dy.at( 0, 0, 0 ) = -0.5;
        dy.at( 0, 2, 0 ) =  0.5;
        return dy;
    }
    inline static Kernel3D<T> dz()
    {
        smart_assert( typeid(T)==typeid(float) || typeid(T)==typeid(double),
                      "Can only use float or double" );
        Kernel3D<T> dz( cv::Vec3i(1,1,3) );
        dz.at( 0, 0, 0 ) = -0.5;
        dz.at( 0, 0, 2 ) =  0.5;
        return dz;
    }

};



template<typename T>
Kernel3D<T>::Kernel3D( const cv::Vec3i& n_size )
{
    reset( n_size );
}


template<typename T>
void Kernel3D<T>::reset( const cv::Vec3i& n_size, cv::Scalar scalar )
{
    Data3D<T>::reset( n_size );

    for( int i=0; i<3; i++ )
    {
        center[i] = n_size[i]/2;
        min_p[i] = -center[i];
        max_p[i] = ( n_size[i]%2!=0 ) ? ( center[i]+1 ) : ( center[i] );
    }
}


template <typename U>
std::ostream& operator<<(std::ostream& out, const Kernel3D<U>& data)
{
    // diable output if the size is too big
    for( int i=0; i<3; i++ )
    {
        if ( data.get_size(i)>9 )
        {
            std::cout << "I am so sorry. data size is too big to display." << std::endl;
            return out;
        }
    }
    // output data
    int x, y, z;
    for ( z=0; z<data.get_size(2); z++ )
    {
        out << "Level " << z << std::endl;
        for ( y=0; y<data.get_size(1); y++ )
        {
            std::cout << "\t";
            for( x=0; x<data.get_size(0); x++ )
            {
                out.precision(3);
                out << std::scientific << data.at( x, y, z ) << " ";
            }
            out << std::endl;
        }
    }
    return out;
}



template<typename T>
Kernel3D<T> Kernel3D<T>::GaussianFilter3D( cv::Vec3i size )
{
    smart_assert( typeid(T)==typeid(float) || typeid(T)==typeid(double),
                  "Can only use float or double" );

    for( int i=0; i<3; i++ )
        smart_assert( size[i]>0 && size[i]%2!=0, "size should be possitive odd number" );

    // Calculate sigma from size:
    //         sigma = 0.3 * ( (ksize-1)/2 - 1 ) + 0.8 = 0.15*ksize + 0.35
    // reference: OpenCv Documentation 2.6.4.0 getGaussianKernel
    // http://docs.opencv.org/modules/imgproc/doc/filtering.html#creategaussianfilter
    // If we calculate sigma based on 99.7% Rule:
    //         sigma = ( size - 1 ) /6 = 0.17 * size - 0.17
    // But we are flowing the definition of OpenCv here
    cv::Mat gaussian[3];
    for( int i=0; i<3; i++ ) gaussian[i] = cv::getGaussianKernel( size[i], 0, CV_64F );

    Kernel3D<T> kernel(size);
    int x, y, z;
    for( z=0; z<kernel.get_size_z(); z++ )
    {
        for( y=0; y<kernel.get_size_y(); y++ )
        {
            for( x=0; x<kernel.get_size_x(); x++ )
            {
                kernel.at(x,y,z)
                    = gaussian[0].at<double>(x)
                      * gaussian[1].at<double>(y)
                      * gaussian[2].at<double>(z);
            }
        }
    }

    return kernel;
}


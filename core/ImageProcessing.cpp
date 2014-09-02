#include "ImageProcessing.h"

#include <queue>

using namespace std;
using namespace cv;

void ImageProcessing::histogram( Data3D<short>& data,
                                 cv::Mat_<double>& range, Mat_<double>& hist, int number_of_bins )
{
    cout << "Calculating Image Histogram" << endl;

    // get the minimum and maximum value in the data
    Vec<short,2> min_max = data.get_min_max_value();
    const double min = min_max[0];
    const double max = min_max[1] + 1;
    const double diff = max - min;

    // set up the range vector
    range = Mat_<double>( number_of_bins, 1, 0.0);
    for( int i=0; i<number_of_bins; i++ )
    {
        range.at<double>(i) = 1.0 * i * diff / number_of_bins + min;
    }

    // set the hist vector
    hist = Mat_<double>( number_of_bins, 1, 0.0);
    Mat_<short>::const_iterator it;
    for( it=data.getMat().begin(); it<data.getMat().end(); it++ )
    {
        const short& value = (*it);
        int pos = int( 1.0 * number_of_bins * (value-min) / diff );
        hist.at<double>( pos )++;
    }
}

Mat ImageProcessing::histogram_with_opencv( Data3D<short>& data, int number_of_bins )
{
    cout << "Calculating Image Histogram using build in OpenCV function" << endl;
    cout << "This function may be fairly memory intensive. Be careful when using." << endl;

    Mat mat = data.getMat();

    mat.convertTo( mat, CV_32F );
    // normalize( mat, mat, 0, number_of_bins, CV_MINMAX );

    // Establish the number of bins
    int histSize = number_of_bins;

    // Set the ranges
    float range[] =  // 0.0f, number_of_bins
    {
        1.0f * std::numeric_limits<short>::min(),
        1.0f * std::numeric_limits<short>::max()
    };//the upper boundary is exclusive
    const float* histRange = { range };

    // Compute the histograms:
    Mat hist;
    bool uniform = true;  // true: bins to have the same size (uniform)

    calcHist( &mat, 1, 0,
              Mat(), // mask
              hist,  // Output Array
              1, &histSize, &histRange, uniform, true );

    // Draw the histogram
    int hist_w = number_of_bins;
    int hist_h = int( number_of_bins * 0.618 );
    int bin_w = cvRound( (double) hist_w/histSize );

    Mat histImage( hist_h, hist_w, CV_8UC3, Scalar(0,0,0) );

    // Normalize the result to [ 0, histImage.rows ]
    normalize( hist, hist, 0, histImage.rows, NORM_MINMAX, -1, Mat() );

    // Draw for each channel
    for( int i = 1; i < histSize; i++ )
    {
        line( histImage, Point( bin_w*(i-1), hist_h - cvRound( hist.at<float>(i-1)) ) ,
              Point( bin_w*(i), hist_h - cvRound( hist.at<float>(i)) ),
              Scalar( 255, 255, 255),
              2,     // thickness
              CV_AA, // antialiased line
              0  );
    }

    /// Display
    namedWindow("Histogram", CV_WINDOW_AUTOSIZE );
    imshow("Histogram", histImage );
    imwrite("Histogram.png", histImage );
    waitKey(0);
    cout << "done. " << endl << endl;

    return hist;
}

void ImageProcessing::histogram_for_slice( Image3D<short>& data )
{
    cout << "Calculating Image Histogram by Slice" << endl;
    const Mat mat = data.getMat();

    vector<double> intensities( data.get_size_z() );
    double max_intensity;
    double min_intensity;
    intensities[0] = max_intensity = min_intensity = cv::sum( mat.row(0) )[0];

    for( int i=1; i<data.get_size_z(); i++ )
    {
        intensities[i] = cv::sum( mat.row(i) )[0];
        max_intensity = std::max( intensities[i], max_intensity );
        min_intensity = std::min( intensities[i], min_intensity );
    }

    int w = data.get_size_z();
    int h = int( data.get_size_z() * 0.618 );

    Mat histImage( h, w, CV_8UC3, Scalar(0,0,0) );
    for( int i=1; i<data.get_size_z(); i++ )
    {
        line( histImage,
              Point( i-1, int( h*intensities[i-1]/max_intensity ) ),
              Point( i,   int( h*intensities[i]/max_intensity ) ),
              Scalar( 255, 255, 255),
              2, // thickness
              8, // antialiased line
              0  );
    }

    namedWindow("Histogram of Slice", CV_WINDOW_AUTOSIZE );
    imshow("Histogram of Slice", histImage );
    imwrite("Histogram of Slice.png", histImage );
    waitKey(0);
}








void ImageProcessing::dilate( Data3D<unsigned char>& src, const int& ks )
{
    Data3D<unsigned char> temp( src.get_size() );



    int counter = 0;
    cout << endl;

    #pragma omp parallel for
    for( int z=0; z<src.SZ(); z++ )
    {
        int x, y, i, j, k;

        for( y=0; y<src.SY(); y++ )
        {
            for( x=0; x<src.SX(); x++ )
            {
                if( !src.at(x,y,z) ) continue;

                // dialte around the voxel
                for( i=-ks; i<=ks; i++ )
                {
                    for( j=-ks; j<=ks; j++ )
                    {
                        for( k=-ks; k<=ks; k++ )
                        {
                            if( temp.isValid(x+i,y+j,z+k) ) {
                                temp.at(x+i,y+j,z+k) = 255;
                            }
                        }
                    }
                }
            }
        }
    }
    src = temp;
}


void ImageProcessing::erose( Data3D<unsigned char>& src, const int& ks )
{
    Data3D<unsigned char> temp( src.get_size(), 255 );
    int x,y,z;
    int i,j,k;
    for( z=0; z<src.SZ(); z++ ) for( y=0; y<src.SY(); y++ ) for( x=0; x<src.SX(); x++ )
            {
                // look around the voxel
                for( i=-ks; i<=ks; i++ ) for( j=-ks; j<=ks; j++ ) for( k=-ks; k<=ks; k++ )
                        {
                            if( src.isValid(x+i,y+j,z+k) )
                            {
                                if( src.at( x+i,y+j,z+k )!=255 ) temp.at(x,y,z) =  0;
                            }
                            else
                            {
                                temp.at(x,y,z) = 0;
                            }
                        }
            }
    src = temp;
}

void ImageProcessing::closing( Data3D<unsigned char>& src, const int& ks )
{
    IP::dilate( src, 2 );
    IP::erose( src, 2 );
}

#include "CVPlot.h"

#include <istream>
#include <opencv2/highgui/highgui.hpp> // for drawing lines and visualization
#include "smart_assert.h"

using namespace std;
using namespace cv;

namespace CVPlot
{

bool draw( const string& name, vector<vector<double> >& funcs,
           unsigned im_height,        // image height
           unsigned im_width )           // image width, will be computed based on the size of mat_ys if left empty

{
    // Maximum number of groups supported by the function
    static const int MAX_GROUP = 6;
    // Set the color the lines for each group
    static cv::Scalar colors[MAX_GROUP] =
    {
        Scalar(  0,   0, 255), // red
        Scalar(  0, 155,   0), // green
        Scalar(255,   0,   0), // blue
        Scalar(  0,   0, 255), // red
        Scalar(255,   0, 255),
        Scalar(255, 255,   0)
    };
    // magin of the plot
    static const float margin = 0.05f;

    const unsigned numY = funcs.size();
    smart_return( numY>0, "Data mat_ys should not be enmpty. ", false );
    smart_return( numY<=MAX_GROUP, "Eceeed max num of functions. ", false );

    const unsigned numX = funcs[0].size();
    for( unsigned i=0; i<numY; i++ )
    {
        smart_return( funcs[i].size()==numX, "vector should have same size. ", false );
    }

    if( im_width==0 ) im_width = std::max(numX, (unsigned) 400);

    double scale = (double) im_width / numX;

    // find the min max value in all the functions
    double minVal = funcs[0][0];
    double maxVal = funcs[0][0];
    for( unsigned i=0; i<numY; i++ ) for( unsigned j=0; j<numX; j++ )
        {
            maxVal = std::max( maxVal, funcs[i][j] );
            minVal = std::min( minVal, funcs[i][j] );
        }
    const double max_min_gap = maxVal - minVal;

    // draw the plot on a mat
    Mat im_bg( im_height, im_width, CV_8UC3,
               /*Default Background Color*/ Scalar(155, 255, 155) );

    Mat im_result = im_bg.clone();
    for( unsigned it = 0; it < funcs.size(); it++ )
    {
        // draw the image a number of times for color blending
        Mat temp = im_bg.clone();
        for( unsigned int it2 = 0; it2 < funcs.size(); it2++ )
        {
            unsigned i = (it+it2) % numY;
            for( unsigned j=1; j < numX; j++ )
            {
                double v1 = funcs[i][j-1];
                double v2 = funcs[i][j];
                Point p1, p2;
                p1.x = int( (j-1) * scale );
                p1.y = int( im_height * ( margin + (1-2*margin)*(1.0 - (v1-minVal)/max_min_gap ) ) );
                p2.x = int( j * scale );
                p2.y = int( im_height * ( margin + (1-2*margin)*(1.0 - (v2-minVal)/max_min_gap ) ) );

                cv::line( temp, p1, p2, colors[i], 1.3, CV_AA );
            }
        }
        // color blending
        double weight = 1.0 * it / numY;
        cv::addWeighted(im_result, weight, temp, 1 - weight, 0, im_result);
    }

    // show result in window and save to file
    if( name.find('.')!=std::string::npos )
        cv::imwrite( "output/" + name, im_result );
    else
        cv::imwrite( "output/" + name + ".png", im_result );

//    cv::imshow( name.c_str(), im_result );
//    cv::waitKey(0);

    return true;
}
}


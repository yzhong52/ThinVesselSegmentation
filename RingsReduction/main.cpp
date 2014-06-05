#include <iostream>

#include "Data3D.h"
#include "GLViwerCore.h"


//#include <opencv2/opencv.hpp>

using namespace std;
using namespace cv;
int main()
{
   // laoding data
    Data3D<short> im_short;
    bool flag = im_short.load( "../data/data15.data" );
    if( !flag ) return 0;

    im_short.show();

    GLViewerCore viewer;
    viewer.addObject( im_short,  GLViewer::Volumn::MIP );
    viewer.go(400, 400, 1);
}


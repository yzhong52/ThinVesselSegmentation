#include <iostream>
#include "Data3D.h"
#include "GLViewerCore.h"
#include "CVPlot.h"

using namespace std;
using namespace cv;

int main( void )
{
    // draw a plot using CVPLOT
    /*
    vector<double> func1;
    func1.push_back(5);
    func1.push_back(4);
    func1.push_back(3);
    vector<double> func2;
    func2.push_back(4);
    func2.push_back(4);
    func2.push_back(5);
    vector<vector<double> > funcs;
    funcs.push_back( func1 );
    funcs.push_back( func2 );

    CVPlot::draw( "ddldl", funcs );
    return 0;
    /**/

    // Loading data
    Data3D<short> im_short;
    bool flag = im_short.load( "../data/data15.data" );
    if( !flag ) return 0;

    GLViewerCore viewer;
    viewer.addObject( im_short,  GLViewer::Volumn::MIP );
    viewer.display(400, 400, 1);
}

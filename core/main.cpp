#include <iostream>
#include "Data3D.h"
#include "GLViwerCore.h"

using namespace std;

int main( void )
{
    // laoding data
    Data3D<short> im_short;
    bool flag = im_short.load( "../data/data15.data" );
    if( !flag ) return 0;

    GLViewerCore viewer;
    viewer.addObject( im_short,  GLViewer::Volumn::MIP );
    viewer.display(400, 400, 1);
}

#include <iostream>

#include "Data3D.h"
#include "GLViwerCore.h"
#include "RingsReduction.h"

using namespace std;
using namespace cv;




int main()
{
   // laoding data
    Data3D<short> im_short;
    bool flag = im_short.load( "../temp/vessel3d.data", Vec3i(585, 525, 10), true, true );
    if( !flag ) return 0;

    Data3D<short> im_rd;
    RR::a_dummy_method( im_short, im_rd );

    im_short.show("Before Redution", 5 );
    im_rd.show( "After Reduction", 5 );
}


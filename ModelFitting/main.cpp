#define _USE_MATH_DEFINES
#include <math.h>
#include <iomanip> // For multi-threading

#include <opencv2/core/core.hpp>
#include <assert.h>
#include <iostream>
#include <limits>
#include <thread> // C++11

#include "Line3D.h"
#include "Data3D.h"
#include "Line3DTwoPoint.h"
#include "LevenbergMarquardt.h"
#include "SyntheticData.h"
#include "ImageProcessing.h"
#include "Timer.h"
#include "Neighbour26.h"
#include "SparseMatrix/SparseMatrix.h"
#include "ModelSet.h"
#include "init_models.h"
#include "GLLineModel.h"
#include "serializer.h"
#include "GLViwerModel.h"

using namespace std;

const double DATA_COST = 1.0;
const double PAIRWISE_SMOOTH = 7.0;
const double DATA_COST2 = DATA_COST * DATA_COST;
const double PAIRWISE_SMOOTH2 = PAIRWISE_SMOOTH * PAIRWISE_SMOOTH;


GLViwerModel vis;

// thread function for rendering
void visualization_func( int numViewports )
{
    vis.display( 1280, 768, numViewports );
}

// function template default type is only available in C++11
template<class T1, class T2=char, class T3=char>
std::thread initViwer( const vector<cv::Vec3i>& dataPoints,
                       const vector<Line3D*>& lines, const vector<int>& labelings,
                       const Data3D<T1>* im1,
                       const Data3D<T2>* im2 = nullptr,
                       const Data3D<T3>* im3 = nullptr )
{
    vis.addObject( *im1 );

    if( im2 ) vis.addObject( *im2 );
    if( im3 ) vis.addObject( *im3 );

    GLViewer::GLLineModel *model = new GLViewer::GLLineModel( im1->get_size() );
    model->updatePoints( dataPoints );
    model->updateModel( lines, labelings );
    vis.objs.push_back( model );

    // compute the number of view ports to use
    int numViewports = 2;
    numViewports += (im2!=nullptr);
    numViewports += (im3!=nullptr);

    // create a thread for visualization
    return std::thread( visualization_func, numViewports );
}

namespace experiments
{
void start_levernberg_marquart( const string& dataname = "data15", bool isDisplay = false )
{
    // Vesselness measure with sigma
    Image3D<Vesselness_Sig> vn_et_sig;
    vn_et_sig.load( dataname + ".et.vn_sig" );
    vn_et_sig.remove_margin_to( Vec3i(50,50,50) );

    // threshold the data and put the data points into a vector
    ModelSet model;
    // model.init_one_model_per_point( vn_et_sig );
    model.deserialize( "data15" );

    cout << "Number of data points: " << model.get_data_size() << endl;

    std::thread visualization_thread;
    if( isDisplay )
    {
        // load original data and vesselness data for rendering
        //Data3D<short> im_short( dataname + ".data" );
        //Data3D<Vesselness_Sig> vn_sig( dataname + ".vn_sig" );
        visualization_thread = initViwer( model.tildaP, model.lines, model.labelID, &vn_et_sig );
    }


    // Levenberg Marquardt
    LevenbergMarquardt lm( model.tildaP, model.labelID, model, model.labelID3d );
    lm.reestimate( 4000, LevenbergMarquardt::Quadratic, dataname );

    model.serialize( "data15" );

    visualization_thread.join();
    // code after this line won't be executed because 'exit(0)' is executed by glut
}
}



int main(int argc, char* argv[])
{
    Mat temp = Mat(200, 200, CV_8UC3);
    cv::imshow( "", temp );

    experiments::start_levernberg_marquart("../temp/data15", true);
    return 0;
}








//// TODO: not compatible with MinGW?
//CreateDirectory(L"./output", NULL);

//////////////////////////////////////////////////
// Loading serialized data
//////////////////////////////////////////////////
//model.deserialize<Line3DTwoPoint>( "output/Line3DTwoPoint.model" );
//if( lines.size()!=dataPoints.size() ) {
//	cout << "Number of models is not corret. " << endl;
//	cout << "Probably because of errors while deserializing the data. " << endl;
//	return 0;
//}

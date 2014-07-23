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
#include "init_models.h" // TODO: remove this!
#include "GLLineModel.h"
#include "serializer.h"
#include "GLViwerModel.h"
#include "make_dir.h"

using namespace std;

const double DATA_COST = 1.0;
const double PAIRWISE_SMOOTH = 7.0;
const double DATA_COST2 = DATA_COST * DATA_COST;
const double PAIRWISE_SMOOTH2 = PAIRWISE_SMOOTH * PAIRWISE_SMOOTH;


namespace experiments
{
void start_levernberg_marquart( const string& dataname = "../temp/data15",
                                const bool& isDisplay = false );
}

int main(int argc, char* argv[])
{
    make_dir( "../temp" );

    // Force the linking of OpenCV
    Mat temp = Mat(10, 10, CV_8UC3);
    cv::imshow( "", temp );

    experiments::start_levernberg_marquart("../temp/data15", true );
    // experiments::view_modelsets();

    return 0;
}



GLViwerModel vis;

// thread function for rendering
void visualization_func( int numViewports )
{
    vis.display( 1280, 768, numViewports );
}

// function template default type is only available in C++11
template<class T1=char, class T2=char, class T3=char>
std::thread initViwer( const ModelSet& modelset,
                       const Data3D<T1>* im1 = nullptr,
                       const Data3D<T2>* im2 = nullptr,
                       const Data3D<T3>* im3 = nullptr )
{
    if( im2 ) vis.addObject( *im1 );
    if( im2 ) vis.addObject( *im2 );
    if( im3 ) vis.addObject( *im3 );

    GLViewer::GLLineModel *model = new GLViewer::GLLineModel( modelset.labelID3d.get_size() );
    model->updatePoints( modelset.tildaP );
    model->updateModel( modelset.lines, modelset.labelID );
    vis.objs.push_back( model );

    // compute the number of view ports to use
    int numViewports = 1;
    numViewports += (im1!=nullptr);
    numViewports += (im2!=nullptr);
    numViewports += (im3!=nullptr);

    // create a thread for visualization
    return std::thread( visualization_func, numViewports );
}

namespace experiments
{
void start_levernberg_marquart( const string& dataname,
                                const bool& isDisplay )
{
    const string datafile = dataname;

    // Vesselness measure with sigma
    Image3D<Vesselness_Sig> vn_et_sig;
    vn_et_sig.load( datafile + ".et.vn_sig" );
    // vn_et_sig.remove_margin_to( Vec3i(30, 40, 30) );
    // vn_et_sig.remove_margin_to( Vec3i(585, 525, 10) );

    stringstream serialized_datafile_stream;
    serialized_datafile_stream << datafile << "_";
    serialized_datafile_stream << vn_et_sig.SX() << "_";
    serialized_datafile_stream << vn_et_sig.SY() << "_";
    serialized_datafile_stream << vn_et_sig.SZ();
    const string serialized_dataname = serialized_datafile_stream.str();

    // threshold the data and put the data points into a vector
    ModelSet model;
    bool flag = model.deserialize( serialized_dataname );
    if( !flag ) model.init_one_model_per_point( vn_et_sig );


    cout << "Number of data points: " << model.tildaP.size() << endl;

    std::thread visualization_thread;
    if( isDisplay )
    {
        // load original data and vesselness data for rendering
        //Data3D<short> im_short( datafile + ".data" );
        //Data3D<Vesselness_Sig> vn_sig( datafile + ".vn_sig" );
        visualization_thread = initViwer( model, &vn_et_sig );
    }

    // Levenberg Marquardt
    LevenbergMarquardt lm( model.tildaP, model.labelID, model, model.labelID3d );
    lm.reestimate( 400, LevenbergMarquardt::Quadratic, serialized_dataname );

    if( visualization_thread.joinable() ) visualization_thread.join();
    // code after this line won't be executed because 'exit(0)' is executed by glut
}


void view_modelsets( const string& serialized_dataname = "vessel3d_rd_585_525_300" )
{
    // threshold the data and put the data points into a vector
    ModelSet model;
    bool flag = model.deserialize( serialized_dataname );
    if( !flag ) return;

    std::thread visualization_thread;

    visualization_thread = initViwer( model );

    if( visualization_thread.joinable() ) visualization_thread.join();
    // code after this line won't be executed because 'exit(0)' is executed by glut
}

}



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
const double PAIRWISE_SMOOTH = 20.0;
const double DATA_COST2 = DATA_COST * DATA_COST;
const double PAIRWISE_SMOOTH2 = PAIRWISE_SMOOTH * PAIRWISE_SMOOTH;


namespace experiments
{
void start_levenberg_marquardt( const string& dataname = "../temp/data15",
                                const bool& isThined = true,
                                const bool& isDisplay = false,
                                Vec3i crop_size = Vec3i(-1,-1,-1) );

void show_levenberg_marquardt_result( const string& serialized_dataname );
}

int main(int argc, char* argv[])
{
    make_dir( "../temp" );

    // Force the linking of OpenCV
    Mat temp = Mat(10, 10, CV_8UC3);
    cv::imshow( "", temp );

    // default settings
    string dataname = "../temp/data15";
    bool runLevenbergMarquardt = true;
    bool isThined = true;
    bool isDisplay = true;

    // Update settings from 'arguments.txt' file
    ifstream arguments( "arguments.txt" );
    if( arguments.is_open() )
    {
        string temp;

        arguments >> temp;
        if( temp=="-dataname")
        {
            arguments.get(); // remove a white space
            std::getline( arguments, dataname );
        }

        arguments >> temp;
        if( temp=="-runLevenbergMarquardt")
        {
            arguments >> runLevenbergMarquardt;
        }

        arguments >> temp;
        if( temp=="-isThined")
        {
            arguments >> isThined;
        }

        arguments >> temp;
        if( temp=="-isDisplay")
        {
            arguments >> isDisplay;
        }
    }


    if( runLevenbergMarquardt ){
        // Run Levenberg marquardt algorithm
        experiments::start_levenberg_marquardt( dataname, isThined, isDisplay );
    } else {
        // Display result of Levenberg Marquardt only
        experiments::show_levenberg_marquardt_result( dataname );
    }

    return 0;
}


////////////////////////////////////////////////////////////////////////////////////////////
// For visualization
////////////////////////////////////////////////////////////////////////////////////////////

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
    if( im1 ) vis.addObject( *im1 );
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



////////////////////////////////////////////////////////////////////////////////////////////
// Experiments
////////////////////////////////////////////////////////////////////////////////////////////

namespace experiments
{
void start_levenberg_marquardt( const string& dataname,
                                const bool& isThined,
                                const bool& isDisplay,
                                Vec3i crop_size )
{
    const string datafile = dataname;

    // Vesselness measure with sigma
    Image3D<Vesselness_Sig> vn_sig;
    vn_sig.load( datafile + (isThined?".et":"") + ".vn_sig" );

    if( crop_size[0]>0 )
    {
        crop_size[0] = std::min( crop_size[0], vn_sig.SX() );
        crop_size[1] = std::min( crop_size[1], vn_sig.SY() );
        crop_size[2] = std::min( crop_size[2], vn_sig.SZ() );
        vn_sig.remove_margin_to( crop_size );
    }

    stringstream serialized_datafile_stream;
    serialized_datafile_stream << datafile << "_";
    serialized_datafile_stream << vn_sig.SX() << "_";
    serialized_datafile_stream << vn_sig.SY() << "_";
    serialized_datafile_stream << vn_sig.SZ();
    const string serialized_dataname = serialized_datafile_stream.str();

    // threshold the data and put the data points into a vector
    ModelSet model;
    bool flag = model.deserialize( serialized_dataname );
    if( !flag ) model.init_one_model_per_point( vn_sig );


    cout << "Number of data points: " << model.tildaP.size() << endl;

    std::thread visualization_thread;
    if( isDisplay )
    {
        //Load original data and vesselness data for rendering
        // Data3D<short> im_short( datafile + ".data" );
        //Data3D<Vesselness_Sig> vn_sig( datafile + ".vn_sig" );
        visualization_thread = initViwer( model, &vn_sig );
    }

    // Levenberg Marquardt
    LevenbergMarquardt lm( model.tildaP, model.labelID, model, model.labelID3d );
    lm.reestimate( 400, LevenbergMarquardt::Quadratic, serialized_dataname );

    if( visualization_thread.joinable() ) visualization_thread.join();
    // code after this line won't be executed because 'exit(0)' is executed by glut
}


void show_levenberg_marquardt_result( const string& serialized_dataname )
{
    // threshold the data and put the data points into a vector
    ModelSet model;
    bool flag = model.deserialize( serialized_dataname );
    if( !flag ) return;

    std::thread visualization_thread = initViwer( model );

    if( visualization_thread.joinable() ) visualization_thread.join();
    // code after this line won't be executed because 'exit(0)' is executed by glut
}

}



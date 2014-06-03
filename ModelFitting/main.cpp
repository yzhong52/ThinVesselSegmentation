// ModelFitting.cpp : Defines the entry point for the console application.
//
#define _USE_MATH_DEFINES
#include <math.h>
#include "Line3D.h"

// For multithreading
#include <iomanip>
#include <opencv2/core/core.hpp>
#include <assert.h>
#include <iostream>
#include <limits>

// This project is build after VesselNess.
// Some of the building blocks (Data3D, Visualization) are borrowed from VesselNess.
#include "Data3D.h"

#include "MinSpanTree.h"

#include "Line3DTwoPoint.h"
#include "LevenburgMaquart.h"
#include "SyntheticData.h"
#include "ImageProcessing.h"
#include "Timer.h"
#include "Neighbour26.h"
#include "SparseMatrix/SparseMatrix.h"
#include "ModelSet.h"
#include "init_models.h"
#include "GLLineModel.h"
#include <thread> // C++11

using namespace std;

const double DATA_COST = 1.0;
const double PAIRWISE_SMOOTH = 7.0;
const double DATA_COST2 = DATA_COST * DATA_COST;
const double PAIRWISE_SMOOTH2 = PAIRWISE_SMOOTH * PAIRWISE_SMOOTH;

// thead for visualization
std::thread visualization_thread;

//#define IS_VISUAL
#ifdef IS_VISUAL // add visualization model
#include "GLViwerModel.h"

GLViwerModel vis;
// thread function for rendering
void visualization_func( int numViewports )
{
    vis.go( 1280, 768, numViewports );
}

// function template default type C++11
template<class T1, class T2=char, class T3=char>
void initViwer( const vector<cv::Vec3i>& dataPoints,
                const vector<Line3D*>& lines, const vector<int>& labelings,
                const Data3D<T1>* im1,
                const Data3D<T2>* im2 = nullptr,
                const Data3D<T3>* im3 = nullptr)
{
    vis.addObject( *im1 );

    if( im2 ) vis.addObject( *im2 );
    if( im3 ) vis.addObject( *im3 );

    GLViewer::GLLineModel *model = new GLViewer::GLLineModel( im1->get_size() );
    model->updatePoints( dataPoints );
    model->updateModel( lines, labelings );
    vis.objs.push_back( model );

    int numViewports = 2;
    numViewports += (im2!=nullptr);
    numViewports += (im3!=nullptr);

    // create visualiztion thread
    visualization_thread = std::thread(visualization_func, numViewports );

}

#else

// function template default type C++11
template<class T1, class T2=char, class T3=char>
void initViwer( const vector<cv::Vec3i>& dataPoints,
                const vector<Line3D*>& lines, const vector<int>& labelings,
                const Data3D<T1>* im1,
                const Data3D<T2>* im2 = nullptr,
                const Data3D<T3>* im3 = nullptr ){ }
#endif

namespace experiments
{

void start_levernberg_marquart( const string& dataname = "data15", bool isDisplay = false )
{
    // Vesselness measure with sigma
    Image3D<Vesselness_Sig> vn_et_sig;
    vn_et_sig.load( dataname + ".et.vn_sig" );

    // threshold the data and put the data points into a vector
    Data3D<int> labelID3d;
    vector<cv::Vec3i> tildaP;
    ModelSet<Line3D> model;
    vector<int> labelID;
    each_model_per_point( vn_et_sig, labelID3d, tildaP, model, labelID );
    cout << "Number of data points: " << tildaP.size() << endl;

    if( isDisplay ){
        // create a thread for rendering
        Data3D<Vesselness_Sig> vn_sig( dataname + ".vn_sig" );
        Data3D<short> im_short( dataname + ".data" );
        initViwer( tildaP, model.models, labelID, &im_short, &vn_sig, &vn_et_sig );
    }

    // Levenberg-Marquart
    LevenburgMaquart lm( tildaP, labelID, model, labelID3d );
    lm.reestimate( 4000, LevenburgMaquart::Quadratic, dataname );
}
}


int main(int argc, char* argv[])
{
    experiments::start_levernberg_marquart("data15", true);

    cout << "Main Thread is Done. " << endl;
    visualization_thread.join();
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

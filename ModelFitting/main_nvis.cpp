// ModelFitting.cpp : Defines the entry point for the console application.
//
#define _USE_MATH_DEFINES
#include <math.h>
#include "Line3D.h"


#include <iomanip> // For multithreading

#include <opencv2/core/core.hpp>
#include <assert.h>
#include <iostream>
#include <limits>
#include <thread> // C++11

#include "Data3D.h"
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
#include "VesselnessTypes.h"

using namespace std;

const double DATA_COST = 1.0;
const double PAIRWISE_SMOOTH = 7.0;
const double DATA_COST2 = DATA_COST * DATA_COST;
const double PAIRWISE_SMOOTH2 = PAIRWISE_SMOOTH * PAIRWISE_SMOOTH;

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

    // Levenberg-Marquart
    LevenburgMaquart lm( tildaP, labelID, model, labelID3d );
    lm.reestimate( 4000, LevenburgMaquart::Quadratic, dataname );
}
}


int main(int argc, char* argv[])
{
    Mat temp = Mat(200, 200, CV_8UC3);
    cv::imshow( "", temp );

    experiments::start_levernberg_marquart("data15", true);

    cout << "Main Thread is Done. " << endl;
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

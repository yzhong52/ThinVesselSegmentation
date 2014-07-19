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
#include "make_dir.h"
#include "send_email.h"


using namespace std;

const double DATA_COST = 1.0;
const double PAIRWISE_SMOOTH = 7.0;
const double DATA_COST2 = DATA_COST * DATA_COST;
const double PAIRWISE_SMOOTH2 = PAIRWISE_SMOOTH * PAIRWISE_SMOOTH;


namespace experiments
{
void start_levernberg_marquart( const string& foldername = "../data",
                                const string& dataname = "data15" )
{
    const string datafile = foldername + dataname;

    // Vesselness measure with sigma
    Image3D<Vesselness_Sig> vn_et_sig;
    vn_et_sig.load( datafile + ".et.vn_sig" );
    // vn_et_sig.remove_margin_to( Vec3i(30, 30, 30) );

    stringstream serialized_datafile_stream;
    serialized_datafile_stream << dataname << "_";
    serialized_datafile_stream << vn_et_sig.SX() << "_";
    serialized_datafile_stream << vn_et_sig.SY() << "_";
    serialized_datafile_stream << vn_et_sig.SZ();
    const string serialized_dataname = serialized_datafile_stream.str();

    // threshold the data and put the data points into a vector
    ModelSet model;
    bool flag = model.deserialize( serialized_dataname );
    if( !flag ) model.init_one_model_per_point( vn_et_sig );

    cout << "Number of data points: " << model.get_data_size() << endl;

    // Levenberg Marquardt
    LevenbergMarquardt lm( model.tildaP, model.labelID, model, model.labelID3d );
    lm.reestimate( 400, LevenbergMarquardt::Quadratic, serialized_dataname );

    model.serialize( serialized_dataname );
}
}



int main(int argc, char* argv[])
{
    

    make_dir( "output" );
    Mat temp = Mat(200, 200, CV_8UC3);
    
    send_email();
    send_email();
    send_email();
    return 0; 

    experiments::start_levernberg_marquart("../temp/", "data15" );
    send_email();
    return 0;
}



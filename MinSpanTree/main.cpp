#include <iostream>
#include <vector>
#include "MSTGraph.h"
#include "MSTEdge.h"
#include "GLMinSpanTree.h"
#include "ComputeMST.h"
#include "MSTEdgeExt.h"

/*Note: A lot of code from ModelFitting is being recompiled and reused here. */
#include "../ModelFitting/ModelSet.h"
#include "../ModelFitting/GLViwerModel.h"
#include "../ModelFitting/Neighbour26.h"

using namespace std;
using namespace MST;
using namespace cv;


int main( int argc , char *argv[] )
{
    // orignal data
    string dataname  = "../temp/data15";
    // models after model fitting
    string modelname = "../temp/data15_134_113_116";
    // a mask to be ignore during mst computation
    string maskname  = "N/A";

    // Update dataname and modelname from 'arguments.txt' file
    ifstream arguments( "arguments.txt" );
    if( arguments.is_open() )
    {
        string temp;
        do
        {
            arguments >> temp;
            if( temp=="-dataname")
            {
                arguments >> dataname;
            }
            else if( temp=="-modelname")
            {
                arguments >> modelname;
            }
            else if ( temp=="-maskname" )
            {
                arguments >> maskname;
            }
        }
        while( !arguments.eof() );
    }


    // TODO: This is for force the linking of OpenCV
    Mat m = Mat(1,1,CV_32F);
    imshow( "Temp", m );

    bool flag = false;

    // For visualization
    GLViwerModel vis;

    // A mask to be ignored while computing minimum spanning tree
    Data3D<unsigned char> mask;
    if( maskname!="N/A" ) mask.load( maskname+".data" );
    //vis.addObject( mask,  GLViewer::Volumn::MIP );

    //*
    Data3D<short> im_short;
    flag = im_short.load( dataname + ".data" );
    if( !flag ) return 0;
    vis.addObject( im_short,  GLViewer::Volumn::MIP );
    im_short.reset(); // releasing memory
    /**/

    //*
    Data3D<Vesselness_Sig> vn_sig;
    flag = vn_sig.load( dataname + ".vn_sig" );
    if( !flag ) return 0;
    vis.addObject( vn_sig,  GLViewer::Volumn::MIP );
    vn_sig.reset(); // releasing memory
    /**/

    // Loading models
    ModelSet modelset;
    if( mask.is_empty() )
    {
        flag = modelset.deserialize( modelname );
    }
    else
    {
        flag = modelset.deserialize( modelname, mask );
    }
    if( !flag ) return 0;
    GLViewer::GLLineModel *model_obj = nullptr;
    model_obj = new GLViewer::GLLineModel( modelset.labelID3d.get_size() );
    model_obj->updatePoints( modelset.tildaP );
    model_obj->updateModel( modelset.lines, modelset.labelID );
    vis.objs.push_back( model_obj );

    //*
    Data3D<Vesselness_Sig> vn_sig_et;
    flag = vn_sig_et.load( dataname + ".et.vn_sig" );
    if( !flag ) return 0;
    //vis.addObject( vn_sig_et,  GLViewer::Volumn::MIP );
    /**/

    /*
    ModelSet modelset_org;
    modelset_org.init_one_model_per_point( vn_sig_et );
    GLViewer::GLLineModel *modelset_org_obj = nullptr;
    modelset_org_obj = new GLViewer::GLLineModel( modelset_org.labelID3d.get_size() );
    modelset_org_obj->updatePoints( modelset_org.tildaP );
    modelset_org_obj->updateModel( modelset_org.lines, modelset_org.labelID );
    vis.objs.push_back( modelset_org_obj );
    /**/


    //*
    Graph<EdgeExt, cv::Vec3d> tree1;
    DisjointSet djs1;
    ComputeMST::neighborhood_graph( modelset, mask, tree1, djs1 );
    GLViewer::GLMinSpanTree *mstobj1 = nullptr;
    mstobj1 = new GLViewer::GLMinSpanTree( tree1, djs1,
                                           modelset.labelID3d.get_size(),
                                           130, 23 );
    vis.objs.push_back( mstobj1 );
    /**/

    vis.display( 900, 680, 4 );

    return 0;
}



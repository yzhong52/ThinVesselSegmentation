#include "ModelSet.h"
#include <opencv2/core/core.hpp>
#include <iostream>
#include <string>

#include "Line3DTwoPoint.h"
#include "Data3D.h"
#include "ImageProcessing.h"
#include "VesselnessTypes.h"

using namespace std;
using namespace cv;

const std::string ModelSet::file_prefix = "output/serialization_";

ModelSet::ModelSet(void)
{

}


ModelSet::~ModelSet(void)
{
    for( unsigned i=0; i<lines.size(); i++ )
    {
        delete lines[i];
    }
}


void ModelSet::serialize( std::string file ) const
{
    // output file stream
    std::ofstream fout( file_prefix + file );

    // serialize lines
    fout << lines.size() << std::endl;
    for( unsigned i=0; i<lines.size(); i++ )
    {
        lines[i]->serialize( fout );
    }


    const int num_points = tildaP.size();
    fout << num_points << std::endl;
    for( int i=0; i<num_points; i++ )
    {
        fout << tildaP[i][0] << ' ';
        fout << tildaP[i][1] << ' ';
        fout << tildaP[i][2] << ' ';

        fout << labelID[i] << std::endl;
    }
    fout.close();

    labelID3d.save( file_prefix + file + ".labelID3d" );
}


bool ModelSet::deserialize( std::string file )
{
    // get the file stream
    std::ifstream fin( file_prefix + file );
    if( !fin.is_open() )
    {
        cout << "The following serialization file is not found: ";
        cout << "'" << file_prefix + file << "'" << endl;
        return false;
    }

    // deserializing lines
    for( unsigned i=0; i<lines.size(); i++ ) delete lines[i];
    lines.clear();
    int num_lines = 0;
    fin >> num_lines;
    for( int i=0; i<num_lines; i++ )
    {
        Line3DTwoPoint* new_model = new Line3DTwoPoint();
        new_model->deserialize( fin );
        lines.push_back( new_model );
    }


    int num_points = 0;
    fin >> num_points;
    tildaP = vector<Vec3i>( num_points, Vec3i(0,0,0) );
    labelID = vector<int>( num_points, 0 );
    for( int i=0; i<num_points; i++ )
    {
        fin >> tildaP[i][0];
        fin >> tildaP[i][1];
        fin >> tildaP[i][2];
        fin >> labelID[i];
    }
    fin.close();

    labelID3d.load( file_prefix + file + ".labelID3d" );
    return true;
}



void ModelSet::init_one_model_per_point( const Data3D<Vesselness_Sig>& vn_sig, const float& threshold )
{
    Data3D<float> vn = vn_sig;
    IP::normalize( vn, 1.0f );

    tildaP.clear();
    labelID.clear();
    lines.clear();
    labelID3d.reset( vn.get_size(), -1 );

    for(int z=0; z<vn.SZ(); z++)
    {
        for ( int y=0; y<vn.SY(); y++)
        {
            for(int x=0; x<vn.SX(); x++)
            {
                if( vn.at(x,y,z) > threshold )   // a thread hold
                {
                    // current line id
                    int lid = (int) lines.size();

                    labelID3d.at(x,y,z) = lid;

                    labelID.push_back( lid );

                    tildaP.push_back( Vec3i(x,y,z) );

                    // Compute the parameters of the line
                    const Vec3d pos(x,y,z);
                    const Vec3d& dir = vn_sig.at(x,y,z).dir;
                    const double& sigma = vn_sig.at(x,y,z).sigma;
                    Line3DTwoPoint *tmp  = new Line3DTwoPoint();
                    tmp->setPositions( pos-dir, pos+dir );
                    tmp->setSigma( sigma );
                    this->lines.push_back( tmp );
                }
            }
        }
    }
}

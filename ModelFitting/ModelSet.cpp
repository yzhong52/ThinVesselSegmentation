#include "ModelSet.h"
#include <opencv2/core/core.hpp>
#include <iostream>
#include <string>

#include "Line3DTwoPoint.h"
#include "ImageProcessing.h"
#include "VesselnessTypes.h"

using namespace std;
using namespace cv;


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
    std::ofstream fout( file + ".modelset");

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

    labelID3d.save( file + ".labelID3d" );
}


bool ModelSet::deserialize( std::string file )
{
    labelID3d.load( file + ".labelID3d" );

    // Get the file stream
    std::string modelset_file = file + ".modelset";
    std::ifstream fin( modelset_file );
    if( !fin.is_open() )
    {
        cout << "The following serialization file is not found: ";
        cout << "'" << modelset_file << "'" << endl;
        return false;
    }

    // Deserializing lines
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
    pointID3d.reset( labelID3d.get_size(), -1 );
    for( int i=0; i<num_points; i++ )
    {
        // The observe postion of the point
        fin >> tildaP[i][0];
        fin >> tildaP[i][1];
        fin >> tildaP[i][2];
        // The labeling of the point
        fin >> labelID[i];

        pointID3d.at( tildaP.back() ) = i;
    }
    fin.close();


    return true;
}

bool ModelSet::deserialize( std::string file, const Data3D<unsigned char>& mask )
{

    labelID3d.load( file + ".labelID3d" );
    smart_assert( labelID3d.get_size()==mask.get_size(), "Size should match. ");
    #pragma omp parallel for
    for( int z = 0; z < mask.SZ(); z++ )
    {
        for( int y = 0; y < mask.SY(); y++ )
        {
            for( int x = 0; x < mask.SX(); x++ )
            {
                if( mask.at(x,y,z) )
                {
                    labelID3d.at(x,y,z) = -1;
                }
            }
        }
    }

    // Get the file stream
    std::string modelset_file = file + ".modelset";
    std::ifstream fin( modelset_file );
    if( !fin.is_open() )
    {
        cout << "The following serialization file is not found: ";
        cout << "'" << modelset_file << "'" << endl;
        return false;
    }

    // Deserializing lines
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

    tildaP.clear();
    labelID.clear();
    pointID3d.reset( labelID3d.get_size(), -1 );
    int num_points = 0;
    fin >> num_points;
    for( int i=0; i<num_points; i++ )
    {
        // The observe postion of the point
        Vec3i p;
        fin >> p[0];
        fin >> p[1];
        fin >> p[2];

        // The labeling of the point
        int l;
        fin >> l;

        if( !mask.at(p) )
        {
            pointID3d.at( p ) = tildaP.size();
            tildaP.push_back( p );
            labelID.push_back( l );
        }
    }
    fin.close();

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

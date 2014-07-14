#include "ModelSet.h"

ModelSet::ModelSet(void)
{

}


ModelSet::~ModelSet(void)
{
    for( unsigned i=0; i<models.size(); i++ )
    {
        delete models[i];
    }
}


void ModelSet::serialize( std::string file ) const
{
    std::ofstream fout( file );
    fout << models.size() << std::endl;
    for( unsigned i=0; i<models.size(); i++ )
    {
        models[i]->serialize( fout );
    }
}


void ModelSet::deserialize( std::string file )
{
    // release previous memory
    for( unsigned i=0; i<models.size(); i++ ) delete models[i];

    models = std::vector<Line3D*>();

    // deserializing data
    std::ifstream fin( file );
    int size = 0;
    fin >> size;
    for( int i=0; i<size; i++ )
    {
        Line3DTwoPoint* new_model = new Line3DTwoPoint();
        new_model->deserialize( fin );
        models.push_back( new_model );
    }
}

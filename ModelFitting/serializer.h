#ifndef SERIALIZER_H
#define SERIALIZER_H

#include <string>
#include <vector>
#include <fstream>

#include "Line3D.h"


class Serializer
{
private:
    Serializer() { }
public:
    template<class T>
    static void save( const std::vector<T>& data, const std::string& filename );

    template<class T>
    static void load( std::vector<T>& data, const std::string& filename );
};


template<class T>
void Serializer::save( const std::vector<T>& data, const std::string& filename )
{
    std::ofstream fout( filename );

    fout << data.size() << endl;

    for( unsigned i=0; i < data.size(); i++ ){
        fout << data[i] << ' ';
    }
}


template<class T>
void Serializer::load( std::vector<T>& data, const std::string& filename )
{
    std::ifstream fin( filename );

    int size;
    fin >> size;

    data = std::vector<T>( size );

    for( unsigned i=0; i < data.size(); i++ ){
        fin >> data[i];
    }
}

////////////////////////////////////////////////////////////////////////
// Serialization of <Line3D*>
////////////////////////////////////////////////////////////////////////
//
//template<>
//void Serializer::save( const std::vector<Line3D*> lines, const std::string& filename ){
//    std::ofstream fout( file );
//    fout << models.size() << std::endl;
//    for( int i=0; i<models.size(); i++ )
//    {
//        models[i]->serialize( fout );
//    }
//}
//
//
//template<>
//void Serializer::load( const std::vector<Line3D*> lines, const std::string& filename ){
//    // release previous memory
//    for( int i=0; i<models.size(); i++ ) delete models[i];
//    models = std::vector<T*>();
//
//    // deserializing data
//    std::ifstream fin( file );
//    int size = 0;
//    fin >> size;
//    for( int i=0; i<size; i++ )
//    {
//        ActualType* new_model = new ActualType();
//        new_model->deserialize( fin );
//        models.push_back( new_model );
//    }
//}


#endif // SERIALIZER_H

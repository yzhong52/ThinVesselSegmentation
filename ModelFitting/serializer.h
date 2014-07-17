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

    for( unsigned i=0; i < data.size(); i++ )
    {
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

    for( unsigned i=0; i < data.size(); i++ )
    {
        fin >> data[i];
    }
}


#endif // SERIALIZER_H

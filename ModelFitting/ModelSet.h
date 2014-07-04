#pragma once
#include <vector>
#include <iostream>
#include <string>
#include <fstream>
#include <vector>

template<class T>
class ModelSet
{
public:
    std::vector<T*> models;

    ModelSet(void);
    virtual ~ModelSet(void);

    /// Serialization
    void serialize( std::string file ) const;
    /// Deserialization - Actual Type is for polymorphism
    template< class ActualType > void deserialize( std::string file );
};



template<class T>
ModelSet<T>::ModelSet(void)
{
}


template<class T>
ModelSet<T>::~ModelSet(void)
{
    for( int i=0; i<models.size(); i++ )
    {
        delete models[i];
    }
}


template<class T>
void ModelSet<T>::serialize( std::string file ) const
{
    std::ofstream fout( file );
    fout << models.size() << std::endl;
    for( int i=0; i<models.size(); i++ )
    {
        models[i]->serialize( fout );
    }
}

template<class T> template< class ActualType>
void ModelSet<T>::deserialize( std::string file )
{
    // release previous memory
    for( int i=0; i<models.size(); i++ ) delete models[i];
    models = std::vector<T*>();

    // deserializing data
    std::ifstream fin( file );
    int size = 0;
    fin >> size;
    for( int i=0; i<size; i++ )
    {
        ActualType* new_model = new ActualType();
        new_model->deserialize( fin );
        models.push_back( new_model );
    }
}

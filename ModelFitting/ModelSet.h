#pragma once
#include <vector>
#include <iostream> 
#include <string>

template<class T>
class ModelSet
{
public:
	std::vector<T*> models; 
	ModelSet(void);
	virtual ~ModelSet(void);

	void serialize( std::string file );

	template< class AcutalType > 
	void deserialize( std::string file ); 
};



template<class T>
ModelSet<T>::ModelSet(void)
{
}


template<class T>
ModelSet<T>::~ModelSet(void)
{
	for( int i=0; i<models.size(); i++ ) {
		delete models[i]; 
	}
}


template<class T>
void ModelSet<T>::serialize( std::string file ){
	std::ofstream fout( file ); 
	fout << models.size() << endl;
	for( int i=0; i<models.size(); i++ ) {
		models[i]->serialize( fout );
	}
}

template<class T> template< class AcutalType>
void ModelSet<T>::deserialize( std::string file ){
	// release previous memroy 
	for( int i=0; i<models.size(); i++ ) delete models[i]; 
	models = vector<T*>(); 

	// deserializing data 
	std::ifstream fin( file ); 
	int size = 0; 
	fin >> size;
	for( int i=0; i<size; i++ ) {
		AcutalType* new_model = new AcutalType(); 
		new_model->deserialize( fin ); 
		models.push_back( new_model );
	}
}
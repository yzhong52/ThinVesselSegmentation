#pragma once

#include "stdafx.h"


struct Vesselness_Data;
struct Vesselness_Sig_Data;
struct Vesselness_Nor_Data;
struct Vesselness_All_Data;

class Vesselness;
class Vesselness_Sig;
class Vesselness_Nor;
class Vesselness_All;

struct Vesselness_Data {
	float rsp; // vesselness response
	Vec3f dir; // vessel dirction	
	Vesselness_Data( float s = 0.0f ) : rsp( s ) { }

	bool operator>( const Vesselness_Data& right ) const { return ( (this->rsp) > right.rsp ); }
	bool operator<( const Vesselness_Data& right ) const { return ( (this->rsp) < right.rsp ); }
}; 

struct Vesselness_Sig_Data : public Vesselness_Data {
	Vesselness_Sig_Data( float s = 0.0f ) : Vesselness_Data ( s ) { } 
	float sigma;	// relative size of the vessel
};

struct Vesselness_Nor_Data : public Vesselness_Data {
	Vec3f normals[2];
};

struct Vesselness_All_Data : public Vesselness_Data {
	float sigma;	// relative size of the vessel
	Vec3f normals[2];
};





class Vesselness : public Vesselness_Data {
public:
	// Size tells us how much number of float we are using in the structure. 
	// The returned value should be: 
	//  4 for Vesselness;
	//  5 for Vesselness_Sig;
	// 10 for Vesselness_Nor;
	// 11 for Vesselness_All;
	// This virtual function here will take 2*4 = 8 bytes of data
	static const int _size = sizeof(Vesselness_Data)/sizeof(float);
	int size(void) const { return _size; }

	const float& operator[]( const int& i ) const {
		smart_return_value( i>=0&&i<_size, "index invalid", *(float*)(this) );
		return *((float*)(this)+i);
	}

	float& operator[]( const int& i ) {
		smart_return_value( i>=0&&i<_size, "index invalid", *(float*)(this) );
		return *((float*)(this)+i);
	}
};


class Vesselness_Nor : public Vesselness_Nor_Data {
public:
	static const int _size = sizeof(Vesselness_Nor_Data)/sizeof(float);

	const float& operator[]( const int& i ) const {
		smart_return_value( i>=0&&i<_size, "index invalid", *(float*)(this) );
		return *((float*)(this)+i);
	}

	float& operator[]( const int& i ) {
		smart_return_value( i>=0&&i<_size, "index invalid", *(float*)(this) );
		return *((float*)(this)+i);
	}
};

class Vesselness_All : public Vesselness_All_Data {
public:
	static const int _size = sizeof(Vesselness_All_Data)/sizeof(float);
	
	Vesselness_All(){}

	Vesselness_All( const Vesselness_Nor& v_Nor, const float& s ) {
		this->dir        = v_Nor.dir;
		this->normals[0] = v_Nor.normals[0];
		this->normals[1] = v_Nor.normals[1];
		this->rsp        = v_Nor.rsp;
		this->sigma      = s;
	}

	const float& operator[]( const int& i ) const {
		smart_return_value( i>=0&&i<_size, "index invalid", *(float*)(this) );
		return *((float*)(this)+i);
	}

	float& operator[]( const int& i ) {
		smart_return_value( i>=0&&i<_size, "index invalid", *(float*)(this) );
		return *((float*)(this)+i);
	}
};


class Vesselness_Sig : public Vesselness_Sig_Data {
	Vesselness_Sig& operator=( const float& s );
public:
	static const int _size = sizeof(Vesselness_Sig_Data)/sizeof(float);

	Vesselness_Sig() { }
	Vesselness_Sig( float s ) : Vesselness_Sig_Data( s ) { } 
	operator Vesselness( ) {
		Vesselness vn;
		vn.rsp = this->rsp;
		vn.dir = this->dir;
		return vn; 
	}
	operator float() { return this->rsp; }

	Vesselness_Sig( const Vesselness_All& src ) {
		this->rsp   = src.rsp;
		this->dir   = src.dir;
		this->sigma = src.sigma;
	} 

	Vesselness_Sig& operator=( const Vesselness_All& src ){
		this->rsp   = src.rsp;
		this->dir   = src.dir;
		this->sigma = src.sigma;
		return (*this);
	}

	const float& operator[]( const int& i ) const {
		smart_return_value( i>=0&&i<_size, "index invalid", *(float*)(this) );
		return *((float*)(this)+i);
	}

	float& operator[]( const int& i ) {
		smart_return_value( i>=0&&i<_size, "index invalid", *(float*)(this) );
		return *((float*)(this)+i);
	}
};

// The namespace cv here is necessary. 
// Reference: http://stackoverflow.com/questions/2282349/specialization-of-templateclass-tp-struct-stdless-in-different-namespace
namespace cv{
	// Yuchen: I should understand these better
	template< > class DataType< Vesselness     > : public DataType< Vec<float, Vesselness::_size> > { };

	template< > class DataType< Vesselness_Sig > : public DataType< Vec<float, Vesselness_Sig::_size> > { };

	template< > class DataType< Vesselness_Nor > : public DataType< Vec<float, Vesselness_Nor::_size> > { };

	template< > class DataType< Vesselness_All > : public DataType< Vec<float, Vesselness_All::_size> > { };
}

// This one below works as well 
// template< > class DataType< Vesselness_Sig  > : public DataType< Vec<float,5> >{
//public:
//    typedef Vesselness_Sig value_type;
//    typedef Vesselness_Sig work_type;
//    typedef float channel_type;
//    typedef value_type vec_type;
//    enum { 
//		generic_type = 0, 
//		depth = DataDepth<channel_type>::value, 
//		channels = 5,
//        fmt = ((channels-1)<<8) + DataDepth<channel_type>::fmt,
//        type = CV_MAKETYPE(depth, channels)
//	};
//};;

/////////////////////////////////////////////////////////////////////////////////////////
// These following are defined in <core.hpp>
//
//template<> class DataType<int>
//{
//public:
//    typedef int value_type;
//    typedef value_type work_type;
//    typedef value_type channel_type;
//    typedef value_type vec_type;
//    enum { generic_type = 0, depth = DataDepth<channel_type>::value, channels = 1,
//           fmt=DataDepth<channel_type>::fmt,
//           type = CV_MAKETYPE(depth, channels) };
//};
//
//template<> class DataType<float>
//{
//public:
//    typedef float value_type;
//    typedef value_type work_type;
//    typedef value_type channel_type;
//    typedef value_type vec_type;
//    enum { generic_type = 0, depth = DataDepth<channel_type>::value, channels = 1,
//           fmt=DataDepth<channel_type>::fmt,
//           type = CV_MAKETYPE(depth, channels) };
//};
//
//template<typename _Tp, int cn> class DataType<Vec<_Tp, cn> >
//{
//public:
//    typedef Vec<_Tp, cn> value_type;
//    typedef Vec<typename DataType<_Tp>::work_type, cn> work_type;
//    typedef _Tp channel_type;
//    typedef value_type vec_type;
//    enum { generic_type = 0, depth = DataDepth<channel_type>::value, channels = cn,
//           fmt = ((channels-1)<<8) + DataDepth<channel_type>::fmt,
//           type = CV_MAKETYPE(depth, channels) };
//};
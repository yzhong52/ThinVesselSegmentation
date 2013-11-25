// stdafx.cpp : source file that includes just the standard includes
// VesselNess.pch will be the pre-compiled header
// stdafx.obj will contain the pre-compiled type information

#include "stdafx.h"
#include "Vesselness.h"

// TODO: reference any additional headers you need in STDAFX.H
// and not in this file


int CV_TYPE(const type_info& type){
	if( type==typeid(short) ) {
		return CV_16S;
	} 
	else if ( type==typeid(int) ){
		return CV_32S;
	} 
	else if ( type==typeid(float) ) {
		return CV_32F;
	} 
	else if ( type==typeid(double) ) {
		return CV_64F;
	} 
	else if ( type==typeid(Vesselness) ) {
		return CV_32FC( Vesselness::_size );
	} 
	else if ( type==typeid(Vesselness_Sig) ) {
		return CV_32FC( Vesselness_Sig::_size );
	}
	else if ( type==typeid(Vesselness_Nor) ) {
		return CV_32FC( Vesselness_Nor::_size );
	} 
	else if ( type==typeid(Vesselness_All) ) {
		return CV_32FC( Vesselness_All::_size );
	}
	else if ( type==typeid(unsigned char) ) {
		return CV_8U;
	}
	else if ( type==typeid(unsigned short) ){
		return CV_16U;
	}
	else {
		smart_assert( 0, "Datatype is not supported.");
		return -1;
	}
}


string STR_TYPE(const type_info& type){
	if( type==typeid(short) ) {
		return "short";
	} 
	else if ( type==typeid(int) ){
		return "int";
	} 
	else if ( type==typeid(float) ) {
		return "float";
	} 
	else if ( type==typeid(double) ) {
		return "double";
	} 
	else if( type==typeid(Vesselness) ) {
		stringstream ss;
		ss << "float," << Vesselness::_size;
		return ss.str();
	}
	else if( type==typeid(Vesselness_Sig) ) {
		stringstream ss;
		ss << "float," << Vesselness_Sig::_size;
		return ss.str();
	}
	else if( type==typeid(Vesselness_Nor) ) {
		stringstream ss;
		ss << "float," << Vesselness_Nor::_size;
		return ss.str();
	}
	else if( type==typeid(Vesselness_All) ) {
		stringstream ss;
		ss << "float," << Vesselness_All::_size;
		return ss.str();
	}
	else if( type==typeid(unsigned char) ) {
		return "unsigned_char";
	} 
	else if( type==typeid(unsigned short) ){
		return "unsigned_short";
	}
	else {
		smart_return_value( 0, "Datatype is not supported.", "(*^__^*)Error!");
	}
}


long long fwrite_big( const void* _Str, size_t _Size, long long _Count, FILE* _File ){
	char* ptr = (char*) _Str;
	long long size = (long long) _Size * (long long) _Count;
	// the page size of a i386 system is 4096 bytes
	const size_t page_size = 1<<12; 
	long long size_write_total = 0;
	size_t size_write;
	for( ; size>page_size; size-=page_size, ptr+=page_size ){
		size_write = fwrite( ptr, 1, page_size, _File);
		size_write_total += size_write; 
		if( size_write < page_size ) return size_write_total; 
	}
	size_write = fwrite( ptr, 1, size, _File);
	size_write_total += size_write; 
	return size_write_total; 
}

long long fread_big( void* _DstBuf, size_t _ElementSize, long long _Count, FILE* _File ) {
	char* ptr = (char*) _DstBuf;
	long long size = (long long) _ElementSize * _Count;
	// the page size of a i386 system is 4096 bytes
	const size_t page_size = 1<<12; 
	long long size_read_total = 0;
	size_t size_read = 0;
	for( ; size>page_size; size-=page_size, ptr+=page_size ){
		size_read = fread( ptr, 1, page_size, _File );
		size_read_total += size_read;
		if( size_read < page_size ) return size_read_total; 
	}
	size_read = fread( ptr, 1, size, _File);
	size_read_total += size_read;
	return size_read_total; 
};
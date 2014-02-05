#include "nstdio.h"

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
		if( size_write < page_size ){
			// this indicates an error while writing 
			return size_write_total; 
		}
	}
	size_write = fwrite( ptr, 1, (size_t) size, _File);
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
		if( size_read < page_size ) {
			// this indicates an error while reading 
			return size_read_total; 
		}
	}
	size_read = fread( ptr, 1, (size_t) size, _File);
	size_read_total += size_read;
	return size_read_total; 
};

#include <stdio.h>

// Read/Writing big block of data
long long fwrite_big( const void* _Str, size_t _Size, long long _Count, FILE* _File );
long long fread_big( void* _DstBuf, size_t _ElementSize, long long _Count, FILE* _File ); 


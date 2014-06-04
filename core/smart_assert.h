/* Smart Assertion
 *  - Show assertion under debug mode
 *  - Print assertion message udner release mode
 *
 * Created by Yuchen on Apr 30th, 2013
 * Last Modified by Yuchen on Jul 2nd, 2013
 *
 * Example:
	Code:
		int i = -1;
		smart_assert(i>=0 && i<3, "Index out of bound" );
	Output:
		Assertion failed: i>0 && i<3
		  Messages: Index out of bound
		  Location: file main.cpp, line 17
 */
#ifndef SMART_ASSERT_H
#define SMART_ASSERT_H
#include <assert.h>
#include <iostream>

// The following color is defined using [ANSI colour codes](http://en.wikipedia.org/wiki/ANSI_escape_code)
#define SMA_RED   "\033[0;31m"
#define SMA_BLACK "\x1b[0;49m"

#define smart_assert( condition, message ) \
	if( !(condition) ) { \
		std::cerr << SMA_RED << "Assertion failed: " << (#condition) << std::endl; \
		std::cerr << "  Messages: " << message << std::endl; \
		std::cerr << "  Location: file "<< __FILE__ << ", line " << __LINE__ << std::endl; \
		std::cerr << SMA_BLACK << std::endl;\
	}

#define smart_return( condition, message, return_value ) \
    smart_assert( condition, message )\
	if( !(condition) ) { \
        return return_value;\
	}

#endif

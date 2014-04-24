#include "Timer.h"

#include <sstream> 
#include <string>
#include <iostream> 
#include <unordered_map>
#include <ctime>

using namespace std;


#ifdef WIN32

#include <Windows.h>

static LARGE_INTEGER frequency; // ticks per second

Timmer::Timmer(){	 
	// get ticks per second
	QueryPerformanceFrequency(&frequency);
}

void Timmer::begin( const std::string& function_name ) {
	std::unordered_map<std::string, Data>::iterator it = instance().datas.find( function_name ); 
	if( it==instance().datas.end() ) {
		instance().datas.insert( pair<std::string, Data>( function_name, Data()) ); 
		it = instance().datas.find( function_name ); 
	}
	
	static LARGE_INTEGER t;
	// start timer
	QueryPerformanceCounter(&t);
	it->second.begin_time = 1000.0 * t.QuadPart /frequency.QuadPart; 
}

void Timmer::end( const std::string& function_name ) {
	std::unordered_map<std::string, Data>::iterator it = instance().datas.find( function_name ); 
	if( it!=instance().datas.end() ) {
		// ticks per second
		static LARGE_INTEGER frequency;        
		// get ticks per second
		QueryPerformanceFrequency(&frequency);

		static LARGE_INTEGER t;
		// start timer
		QueryPerformanceCounter(&t);
		it->second.total_run_time += 1000.0 * t.QuadPart /frequency.QuadPart - it->second.begin_time; 

		it->second.count++; 
	} else {
		std::cerr << "Error: This function '" << function_name << "' is not defined" << std::endl; 
		system( "pause" ); 
	}
}

#else


#include <sys/time.h>

Timmer::Timmer(){
}

void Timmer::begin( const std::string& function_name ) {
	std::unordered_map<std::string, Data>::iterator it = instance().datas.find( function_name );
	if( it==instance().datas.end() ) {
		instance().datas.insert( pair<std::string, Data>( function_name, Data()) );
		it = instance().datas.find( function_name );
	}
	
    // start timer
    static timeval t;
    gettimeofday(&t, NULL);
	it->second.begin_time
        = 1.0 * t.tv_sec * 1000.0      // sec to ms
        + 1.0 * t.tv_usec / 1000.0;    // us to ms
}

void Timmer::end( const std::string& function_name ) {
	std::unordered_map<std::string, Data>::iterator it = instance().datas.find( function_name );
	if( it!=instance().datas.end() ) {
		static timeval t;
        gettimeofday(&t, NULL);
		it->second.total_run_time
            += 1.0 * t.tv_sec * 1000.0      // sec to ms
            + 1.0 * t.tv_usec / 1000.0 - it->second.begin_time;
        
		it->second.count++;
	} else {
		std::cerr << "Error: This function '" << function_name << "' is not defined" << std::endl;
		system( "pause" );
	}
}

#endif



std::string Timmer::summery( void ) {
	std::stringstream ss;

	static const string func_name    = "Function Name";
	static const string total_time   = "Total"; 
	static const string called_times = "Acount"; 
	static const string avg_time     = "Average"; 

	const int func_name_size    = max((int)func_name.length(),    22);
	const int total_time_size   = max((int)total_time.length(),   15);
	const int called_times_size = max((int)called_times.length(), 12);
	const int avg_time_size     = max((int)avg_time.length(),     15);

	ss << "+--------------------" << endl;
	ss << "| Profiling Summery ..." << endl;
	ss << "+---------------------------------------" << endl;
	ss << "| ";
	ss.width( func_name_size );
	ss << std::left << "Function Name";
	ss << " |";
	ss.width( total_time_size );
	ss << std::right << "Total Time";
	ss << " |";
	ss.width( called_times_size );
	ss << std::right << "Be Called";
	ss << " |";
	ss.width( avg_time_size );
	ss << std::right << "Average";
	ss << " |" << std::endl; 
	
	std::unordered_map<std::string, Data>::iterator it = instance().datas.begin(); 
	for( ; it != instance().datas.end(); it++ ) {
		ss << "| "; 
		ss.width( func_name_size );
		ss << std::left << it->first << " |"; 
		ss.width( total_time_size - sizeof("ms") );
		ss << std::right<< it->second.total_run_time << " ms |"; 
		ss.width( called_times_size - sizeof("times") );
		ss << std::right<< it->second.count << " times |"; 
		ss.width( avg_time_size - sizeof("ms") );
		ss << std::right<< it->second.total_run_time / it->second.count << " ms |" << endl; 
	}
	
	return ss.str();
}

#ifndef _TIMMER_H
#define _TIMMER_H

#include <string>
#include <unordered_map>

class Timmer {
public:
	static void begin( const std::string& function_name = "Anonymous Function" );

	static void end( const std::string& function_name   = "Anonymous Function" ); 

	static std::string summery( void ); 

private:
	// Private Functions
	Timmer();
	inline static Timmer& instance(){
		static Timmer t; 
		return t; 
	}
	// Data Structures
	struct Data{
		double total_run_time; 
		double begin_time; 
		int count; 
		Data() : total_run_time( 0.0 ), begin_time( 0.0 ), count(0) { } 
	}; 
	std::unordered_map<std::string, Data> datas; 
}; 

#endif 
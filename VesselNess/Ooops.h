#ifndef _OOOPS_H_
#define _OOOPS_H_

#include <exception>

struct Ooops : std::exception {
	string msg;
	Ooops(const string& str = "Ooops! Something is wrong. ") : msg(str) {}
	const char* what() const { return msg.c_str(); }
};

#endif
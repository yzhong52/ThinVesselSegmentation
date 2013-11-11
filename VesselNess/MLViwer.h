#pragma once
#include "stdafx.h"

namespace MatlabViwer
{
#ifdef WIN86
	void plot( Mat_<double> mx, vector< Mat_<double> > mys );
	void plot( Mat_<double> mx, Mat_<double> my );
	void surf( Mat_<double> matz );
#else
	void plot( Mat_<double> mx, vector< Mat_<double> > mys );
	void plot( Mat_<double> mx, Mat_<double> my );
	void surf( Mat_<double> matz );
#endif
};

namespace MLVier = MatlabViwer;
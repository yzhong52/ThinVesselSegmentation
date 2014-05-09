#pragma once

#include <vector> 
#include "opencv2\core\core.hpp"


class Line3D; 
class Vesselness_Sig; 
template<typename T> class ModelSet;
template<typename T> class Data3D;

void each_model_per_point( 
	const Data3D<Vesselness_Sig>& vn_sig,
	Data3D<int>& labelID3d, 
	std::vector<cv::Vec3i>& tildaP,
	ModelSet<Line3D>& model, 
	std::vector<int>& labelID ); 


void each_model_per_local_maximum( 
	const Data3D<Vesselness_Sig>& vn_sig,
	Data3D<int>& labelID3d, 
	std::vector<cv::Vec3i>& tildaP,
	ModelSet<Line3D>& model, 
	std::vector<int>& labelID ); 

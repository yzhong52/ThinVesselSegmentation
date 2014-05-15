#pragma once

#include <vector>
#include <array> 
#include "opencv2\core\core.hpp"

class Line3D; 
template<class T> 
class Data3D;
class SparseMatrixCV; 

extern const double DATA_COST2; 
extern const double PAIRWISE_SMOOTH2; 

typedef void (*SmoothCostFunc)( \
	const Line3D* line_i, const Line3D* line_j, \
	const cv::Vec3d& pi_tilde, const cv::Vec3d& pj_tilde, \
	double& smooth_cost_i, double& smooth_cost_j, void* func_data ); 


// compute datacost for asigning a data point to a line
double compute_datacost_for_one( const Line3D* line_i, const cv::Vec3d& pi );

// compute smoothcost for a pair of neighbouring pixels
void smoothcost_func_quadratic( 
	const Line3D* line_i, const Line3D* line_j,
	const cv::Vec3d& pi_tilde, const cv::Vec3d& pj_tilde,
	double& smooth_cost_i, double& smooth_cost_j, void* func_data = NULL );

// compute smoothcost for a pair of neighbouring pixels
void smoothcost_func_linear( 
	const Line3D* line_i, const Line3D* line_j,
	const cv::Vec3d& pi_tilde, const cv::Vec3d& pj_tilde,
	double& smooth_cost_i, double& smooth_cost_j, void* func_data = NULL );

// compute total energy: smoothcost + datacost
double compute_energy( 
	const std::vector<cv::Vec3i>& dataPoints,
	const std::vector<int>& labelings, 
	const std::vector<Line3D*>& lines,
	const Data3D<int>& indeces, SmoothCostFunc using_smoothcost_func );

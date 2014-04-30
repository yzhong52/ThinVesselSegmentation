#pragma once

#include <vector>
#include "opencv2\core\core.hpp"

class Line3D; 
template<class T> class Data3D;
class SparseMatrixCV; 

extern const double LOGLIKELIHOOD; 
extern const double PAIRWISESMOOTH; 

// compute datacost for asigning a data point to a line
inline double compute_datacost_for_one( 
	const Line3D* line_i, const cv::Vec3d& pi );

// compute datacost for asigning each data point to its line
double compute_datacost( 
	const std::vector<cv::Vec3i>& dataPoints,
	const std::vector<int>& labelings, 
	const std::vector<Line3D*>& lines );

// compute smoothcost for a pair of neighbouring pixels
void compute_smoothcost_for_pair( 
	const Line3D* line_i, const Line3D* line_j,
	const cv::Vec3d& pi_tilde, const cv::Vec3d& pj_tilde,
	double& smooth_cost_i, double& smooth_cost_j );

// computing smoothcost for all pairs of neighbouring pixels
double compute_smoothcost( 
	const std::vector<cv::Vec3i>& dataPoints,
	const std::vector<int>& labelings, 
	const std::vector<Line3D*>& lines,
	const Data3D<int>& indeces );

// compute total energy: smoothcost + datacost
double compute_energy( 
	const std::vector<cv::Vec3i>& dataPoints,
	const std::vector<int>& labelings, 
	const std::vector<Line3D*>& lines,
	const Data3D<int>& indeces );

//// compute energy matrix for all data costs
//cv::Mat compute_matrix_datacost( 
//	const std::vector<cv::Vec3i>& dataPoints,
//	const std::vector<int>& labelings, 
//	const std::vector<Line3D*>& lines );




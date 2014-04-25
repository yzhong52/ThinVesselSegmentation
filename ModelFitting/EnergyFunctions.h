#pragma once

class Line3D; 
template<class T> class Data3D;
#include <vector>
#include "opencv2\core\core.hpp"

extern const double LOGLIKELIHOOD; 
extern const double PAIRWISESMOOTH; 

inline double compute_energy_datacost_for_one( 
	const Line3D* line_i,
	const cv::Vec3d& pi );

double compute_energy_datacost( 
	const std::vector<cv::Vec3i>& dataPoints,
	const std::vector<int>& labelings, 
	const std::vector<Line3D*>& lines );

void compute_energy_smoothcost_for_pair( 
	const Line3D* line_i,
	const Line3D* line_j,
	const cv::Vec3d& pi_tilde, 
	const cv::Vec3d& pj_tilde,
	double& smooth_cost_i, 
	double& smooth_cost_j );

double compute_energy_smoothcost( 
	const std::vector<cv::Vec3i>& dataPoints,
	const std::vector<int>& labelings, 
	const std::vector<Line3D*>& lines,
	const Data3D<int>& indeces );

double compute_energy( 
	const std::vector<cv::Vec3i>& dataPoints,
	const std::vector<int>& labelings, 
	const std::vector<Line3D*>& lines,
	const Data3D<int>& indeces );

cv::Mat compute_matrix_datacost( 
	const std::vector<cv::Vec3i>& dataPoints,
	const std::vector<int>& labelings, 
	const std::vector<Line3D*>& lines );
#pragma once
#include "SparseMatrixCV\SparseMatrixCV.h" 
#include "opencv2\core\core.hpp"
#include "ModelSet.h" 
using namespace cv; 

class Line3D; 
template <typename T> class Data3D; 
template <typename T> class ModelSet; 

extern const double LOGLIKELIHOOD;
extern const double PAIRWISESMOOTH; 

class LevenburgMaquart
{
	const vector<Vec3i>& tildaP;   // original points
	const vector<int>& labelID;
	const Data3D<int>& labelID3d;

	const ModelSet<Line3D>& modelset;
	const vector<Line3D*>& lines; 
	
	int numParamPerLine; 
	int numParam; 

	vector<Vec3d>  P;              // projection points of original points
	vector<SparseMatrixCV> nablaP; // Jacobian matrix of the porjeciton points 
public:
	LevenburgMaquart( const vector<Vec3i>& dataPoints, const vector<int>& labelings, 
		const ModelSet<Line3D>& modelset, const Data3D<int>& labelIDs ); 

	void reestimate( void ); 
private:
	//// Jacobian Matrix - data cost
	void Jacobian_datacost(
		vector<double>& Jacobian_nzv, 
		vector<int>&    Jacobian_colindx, 
		vector<int>&    Jacobian_rowptr,
		vector<double>& energy_matrix );
	// parallelized of the above function Jacobian_datacost
	void Jacobian_datacost_openmp(
		vector<double>& Jacobian_nzv, 
		vector<int>&    Jacobian_colindx, 
		vector<int>&    Jacobian_rowptr,
		vector<double>& energy_matrix );
	void Jacobian_datacost_thread_func(
		vector<double>& Jacobian_nzv, 
		vector<int>&    Jacobian_colindx,  
		vector<int>&    Jacobian_rowptr, 
		vector<double>& energy_matrix,
		int site); 

	// Jacobian Matrix - smooth cost
	void Jacobian_smoothcost( 
		vector<double>& Jacobian_nzv, 
		vector<int>&    Jacobian_colindx,  
		vector<int>&    Jacobian_rowptr, 
		vector<double>& energy_matrix );
	// Jacobian Matrix - smooth cost
	void Jacobian_smoothcost_openmp( 
		vector<double>& Jacobian_nzv, 
		vector<int>&    Jacobian_colindx,  
		vector<int>&    Jacobian_rowptr, 
		vector<double>& energy_matrix );
	void Jacobian_smoothcost_openmp_critical_section(
		vector<double>& Jacobian_nzv, 
		vector<int>&    Jacobian_colindx,  
		vector<int>&    Jacobian_rowptr, 
		vector<double>& energy_matrix );
	// Jacobian Matrix - smooth cost
	// each site one thread
	void Jacobian_smoothcost_thread_func(
		vector<double>& Jacobian_nzv, 
		vector<int>&    Jacobian_colindx,  
		vector<int>&    Jacobian_rowptr, 
		vector<double>& energy_matrix,
		int site );

private:
	// projections of datapoints & 
	// the Jacobain matrix of the corresponding projection point 
	
	// X1, X2: 3 * 1, two end points of the line
	// nablaX: 3 * 12, 3 non-zero values
	// nablaP: 3 * 12
	void  Jacobian_projection( 
		const cv::Vec3d& X1, const cv::Vec3d& X2,                                    // two end points of a line
		const SparseMatrixCV& nablaX1, const SparseMatrixCV& nablaX2,          // Jacobians of the end points of the line
		const cv::Vec3d& tildeP,           const SparseMatrixCV& nablaTildeP,      // a point, and the Jacobian of the point
		cv::Vec3d& P, SparseMatrixCV& nablaP );

	SparseMatrixCV Jacobian_datacost_for_one( const int& site );

	void Jacobian_smoothcost_for_pair( const int& sitei, const int& sitej, 
		SparseMatrixCV& nabla_smooth_cost_i,
		SparseMatrixCV& nabla_smooth_cost_j );

	// update model according to delta (delta can be consider as the gradient 
	// computed with levenberg marquart 
	void update_lines( const Mat_<double>& delta ); 

	// adjust the end points of the lines so that they don't shirt away from the data 
	void adjust_endpoints( void ); 
};


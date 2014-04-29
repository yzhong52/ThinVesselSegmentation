#pragma once
#include "SparseMatrixCV\SparseMatrixCV.h" 
#include "opencv2\core\core.hpp"
using namespace cv; 

class Line3D; 
template <typename T> class Data3D; 
template <typename T> class ModelSet; 

extern const double LOGLIKELIHOOD;
extern const double PAIRWISESMOOTH; 

class LevenburgMaquart
{
	
	const vector<Vec3i>& dataPoints;
	const vector<int>& labelings;
	const ModelSet<Line3D>& modelset;
	const Data3D<int>& labelIDs;
public:
	LevenburgMaquart( const vector<Vec3i>& dataPoints, const vector<int>& labelings, 
		const ModelSet<Line3D>& modelset, const Data3D<int>& labelIDs ) 
		: dataPoints( dataPoints ), labelings( labelings )
		, modelset( modelset ), labelIDs( labelIDs ) { }

	void reestimate( void ); 
private:
	//// Jacobian Matrix - data cost
	void Jacobian_datacost(
		vector<double>& Jacobian_nzv, 
		vector<int>&    Jacobian_colindx, 
		vector<int>&    Jacobian_rowptr,
		vector<double>& energy_matrix );
	
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

private:
	// projections of datapoints & 
	// the Jacobain matrix of the corresponding projection point 
	vector<Vec3d> P;
	vector<SparseMatrixCV> nablaP;
	
	// X1, X2: 3 * 1, two end points of the line
	// nablaX: 3 * 12, 3 non-zero values
	// nablaP: 3 * 12
	void  Jacobian_projection( 
		const cv::Vec3d& X1, const cv::Vec3d& X2,                                    // two end points of a line
		const SparseMatrixCV& nablaX1, const SparseMatrixCV& nablaX2,          // Jacobians of the end points of the line
		const cv::Vec3d& tildeP,           const SparseMatrixCV& nablaTildeP,      // a point, and the Jacobian of the point
		cv::Vec3d& P, SparseMatrixCV& nablaP );

	SparseMatrixCV Jacobian_datacost_for_one( 
		const Line3D* l, 
		const cv::Vec3d tildeP, int site );

	void Jacobian_smoothcost_for_pair( const Line3D* li, const Line3D* lj, 
		const cv::Vec3d& tildePi, const cv::Vec3d& tildePj,
		SparseMatrixCV& nabla_smooth_cost_i,
		SparseMatrixCV& nabla_smooth_cost_j,
		int sitei_i, int site_j);
};


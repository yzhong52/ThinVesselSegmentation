#pragma once

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
	//void Jacobian_datacost( 
	//	const vector<Vec3i>& dataPoints,
	//	const vector<int>& labelings, 
	//	const ModelSet<Line3D>& modelset, 
	//	const Data3D<int>& indeces,
	//	vector<double>& Jacobian_nzv, 
	//	vector<int>&    Jacobian_colindx, 
	//	vector<int>&    Jacobian_rowptr);
	//// Jacobian Matrix - smooth cost
	//void Jacobian_smoothcost( 
	//	const vector<Vec3i>& dataPoints,
	//	const vector<int>& labelings, 
	//	const ModelSet<Line3D>& modelset, 
	//	const Data3D<int>& indeces,
	//	vector<double>& Jacobian_nzv, 
	//	vector<int>&    Jacobian_colindx, 
	//	vector<int>&    Jacobian_rowptr);
};


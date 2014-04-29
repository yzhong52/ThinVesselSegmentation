#include "EnergyFunctions.h"
#include <vector>
#include "opencv2\core\core.hpp"
#include "Line3D.h"
#include "Neighbour26.h"
#include "Data3D.h" 
#include "Timer.h"
#include "SparseMatrixCV\SparseMatrixCV.h"

using namespace std;
using namespace cv; 

#include "../SparseMatrixCV/SparseMatrixCV.h"
#ifdef _DEBUG
	#pragma comment(lib,"../x64/Debug/SparseMatrixCV.lib")
	#pragma comment(lib,"../x64/Debug/SparseMatrix.lib")
#else
	#pragma comment(lib,"../x64/Release/SparseMatrixCV.lib")
	#pragma comment(lib,"../x64/Release/SparseMatrix.lib")
#endif

inline double compute_datacost_for_one( const Line3D* line_i, const Vec3d& pi )
{
	return LOGLIKELIHOOD * LOGLIKELIHOOD * line_i->loglikelihood( pi );
}

double compute_datacost( 
	const vector<Vec3i>& dataPoints,
	const vector<int>& labelings, 
	const vector<Line3D*>& lines )
{
	double energy = 0; 
	for( int site = 0; site < (int) dataPoints.size(); site++ ) {
		int label = labelings[site];
		energy += compute_datacost_for_one( lines[label], dataPoints[site] ); 
	}
	return energy; 
}


void compute_smoothcost_for_pair( 
	const Line3D* line_i,  const Line3D* line_j,
	const Vec3d& pi_tilde, const Vec3d& pj_tilde,
	double& smooth_cost_i, double& smooth_cost_j )
{
	// single projection
	const Vec3d pi = line_i->projection( pi_tilde ); 
	const Vec3d pj = line_j->projection( pj_tilde ); 

	// double projection
	const Vec3d pi_prime = line_j->projection( pi ); 
	const Vec3d pj_prime = line_i->projection( pj ); 

	// distance vector
	const Vec3d pi_pj       = pi - pj; 
	const Vec3d pi_pi_prime = pi - pi_prime; 
	const Vec3d pj_pj_prime = pj - pj_prime;

	// distance
	double dist_pi_pj  = pi_pj.dot(pi_pj); 
	double dist_pi_pi_prime = pi_pi_prime.dot(pi_pi_prime); 
	double dist_pj_pj_prime = pj_pj_prime.dot(pj_pj_prime); 

	if( dist_pi_pj < 1e-20 ) dist_pi_pj = 1e-20; 
	
	if( dist_pi_pi_prime < 1e-20 ) smooth_cost_i = 0; 
	else smooth_cost_i = PAIRWISESMOOTH * PAIRWISESMOOTH * dist_pi_pi_prime / dist_pi_pj; 
	
	if( dist_pj_pj_prime < 1e-20 ) smooth_cost_j = 0; 
	else smooth_cost_j = PAIRWISESMOOTH * PAIRWISESMOOTH * dist_pj_pj_prime / dist_pi_pj; 
}

double compute_smoothcost( 
	const vector<Vec3i>& dataPoints,
	const vector<int>& labelings, 
	const vector<Line3D*>& lines,
	const Data3D<int>& indeces )
{
	double energy = 0; 

	for( int site = 0; site < dataPoints.size(); site++ ) { // For each data point

		// iterate through all its neighbours
		for( int neibourIndex=0; neibourIndex<13; neibourIndex++ ) { 
			// the neighbour position
			int x, y, z; 
			Neighbour26::getNeigbour( neibourIndex, 
				dataPoints[site][0], dataPoints[site][1], dataPoints[site][2], 
				x, y, z ); 
			if( !indeces.isValid(x,y,z) ) continue; // not a valid position
			                                        // otherwise

			int site2 = indeces.at(x,y,z); 
			if( site2==-1 ) continue ; // not a neighbour
			                           // other wise, found a neighbour

			const int l1 = labelings[site];
			const int l2 = labelings[site2];

			if( l1==l2 ) continue; 

			double energy_smoothness_i = 0, energy_smoothness_j = 0; 
			compute_smoothcost_for_pair( lines[l1], lines[l2], 
				dataPoints[site], dataPoints[site2], 
				energy_smoothness_i, energy_smoothness_j ); 

			energy += energy_smoothness_i + energy_smoothness_j; 
		}
	}
	return energy; 
}

double compute_energy( 
	const vector<Vec3i>& dataPoints,
	const vector<int>& labelings, 
	const vector<Line3D*>& lines,
	const Data3D<int>& indeces )
{
	Timer::begin("Compute Energy");

	double datacost   = compute_datacost( dataPoints, labelings, lines ); 
	double smoothcost = compute_smoothcost( dataPoints, labelings, lines, indeces ); 

	Timer::end("Compute Energy");
	return datacost + smoothcost; 
}

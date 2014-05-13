#include "EnergyFunctions.h"
#include <vector>
#include "opencv2\core\core.hpp"
#include "Line3D.h"
#include "Neighbour26.h"
#include "Data3D.h" 
#include "Timer.h"
#include "../SparseMatrixCV/SparseMatrixCV.h"

#ifdef _DEBUG
	#pragma comment(lib,"../x64/Debug/SparseMatrixCV.lib")
	#pragma comment(lib,"../x64/Debug/SparseMatrix.lib")
#else
	#pragma comment(lib,"../x64/Release/SparseMatrixCV.lib")
	#pragma comment(lib,"../x64/Release/SparseMatrix.lib")
#endif

using namespace std;
using namespace cv; 

const double epsilon_double = 1e-50; 

inline double compute_datacost_for_one( const Line3D* line_i, const Vec3d& pi )
{
	const Vec3d proj_point = line_i->projection( pi );
	const Vec3d dir = proj_point - pi;
	const double dist2 = dir.dot(dir);
	const double sigma2 = line_i->getSigma() * line_i->getSigma(); 
	return LOGLIKELIHOOD * LOGLIKELIHOOD * dist2 / sigma2;
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

	// distances
	const double dist_pi_pi_prime2 = pi_pi_prime.dot(pi_pi_prime); 
	const double dist_pj_pj_prime2 = pj_pj_prime.dot(pj_pj_prime); 
	const double dist_pi_pj2       = max( pi_pj.dot(pi_pj), epsilon_double );

	smooth_cost_i = PAIRWISESMOOTH * PAIRWISESMOOTH * dist_pi_pi_prime2 / dist_pi_pj2; 
	smooth_cost_j = PAIRWISESMOOTH * PAIRWISESMOOTH * dist_pj_pj_prime2 / dist_pi_pj2; 
}


double compute_datacost( 
	const vector<Vec3i>& dataPoints,
	const vector<int>& labelings, 
	const vector<Line3D*>& lines )
{
	double energy = 0; 
	for( int site = 0; site < (int) dataPoints.size(); site++ ) {
		const int& label = labelings[site];
		energy += compute_datacost_for_one( lines[label], dataPoints[site] ); 
	}
	return energy; 
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
			// neighbour position
			Vec3i neig;  
			Neighbour26::getNeigbour( neibourIndex, dataPoints[site], neig ); 
			
			if( !indeces.isValid(neig) ) continue; // not a valid position, otherwise

			const int site2 = indeces.at(neig); 
			if( site2==-1 ) continue ; // not a neighbour, other wise, found a neighbour

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
	double datacost   = compute_datacost( dataPoints, labelings, lines ); 
	double smoothcost = compute_smoothcost( dataPoints, labelings, lines, indeces ); 
	return datacost + smoothcost; 
}

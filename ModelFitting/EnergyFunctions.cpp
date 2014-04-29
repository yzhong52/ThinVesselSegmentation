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

// X1, X2: 3 * 1, two end points of the line
// nablaX: 3 * 12, 3 non-zero values
// nablaP: 3 * 12
void  projection_jacobians( 
	const Vec3d& X1, const Vec3d& X2,                                    // two end points of a line
	const SparseMatrixCV& nablaX1, const SparseMatrixCV& nablaX2,          // Jacobians of the end points of the line
	const Vec3d& tildeP,           const SparseMatrixCV& nablaTildeP,      // a point, and the Jacobian of the point
	Vec3d& P, SparseMatrixCV& nablaP )
{	
	Timer::begin("Projection Jacobians");
	// Assume that: P = T * X1 + (1-T) * X2
	Vec3d X1_X2 = X1 - X2;
	const double A = ( tildeP - X2 ).dot( X1_X2 );
	const double B = ( X1_X2 ).dot( X1_X2 );
	const double T = A / B;

	// Compute the Jacobian matrix for Ai, Bi and Aj, Bj
	const SparseMatrixCV nablaX1_nablaX2 = nablaX1 - nablaX2; 

	const SparseMatrixCV nablaA = X1_X2.t() * SparseMatrixCV( nablaTildeP - nablaX2 ) + (tildeP - X2).t() * nablaX1_nablaX2;
	const SparseMatrixCV nablaB = X1_X2.t() * 2 * nablaX1_nablaX2; 

	// Compute the Jacobian matrix for Ti and Tj
	// 1 * N matrices
	const SparseMatrixCV nablaT = ( nablaA * B - nablaB * A ) / ( B * B );

	// Compute the projection point
	P = T * X1 + (1-T) * X2; 
	
	// And the Jacobian matrix of the projection point
	// 3 * N matrices
	const SparseMatrixCV MX1( X1 );
	const SparseMatrixCV MX2( X2 ); 

	nablaP = X1 * nablaT + nablaX1 * T + nablaX2 * (1-T) - X2 * nablaT;

	Timer::end("Projection Jacobians");
	
}


SparseMatrixCV compute_datacost_derivative( const Line3D* l,  const Vec3d tildeP ) {
	Timer::begin("Datacost Derivative");
	Vec3d X1, X2; 
	l->getEndPoints( X1, X2 );

	static const int indecesM1[][2] = { {0, 0}, {1, 1}, {2, 2} }; 
	static const int indecesM2[][2] = { {0, 3}, {1, 4}, {2, 5} }; 
	static const double values[] = { 1.0, 1.0, 1.0 }; 
	static const SparseMatrixCV nablaX1( 3, 6, indecesM1, values, 3 );
	static const SparseMatrixCV nablaX2( 3, 6, indecesM2, values, 3 );
	
	Vec3d P; 
	SparseMatrixCV nablaP; 
	projection_jacobians( X1, X2, nablaX1, nablaX2, tildeP, SparseMatrixCV(3, 6), P, nablaP );
	
	const Vec3d tildeP_P = Vec3d(tildeP) - Vec3d(P); 
	const double tildaP_P_lenght = max( 1e-20, sqrt( tildeP_P.dot(tildeP_P) ) ); // l->distanceToLine( tildeP );

	const SparseMatrixCV res = ( -1.0 / tildaP_P_lenght * LOGLIKELIHOOD ) * tildeP_P.t() * nablaP ; 

	Timer::end("Datacost Derivative");
	return res; 
}


void compute_smoothcost_derivative( const Line3D* li, const Line3D* lj, 
	const Vec3d& tildePi, const Vec3d& tildePj,
	SparseMatrixCV& nabla_smooth_cost_i, 
	SparseMatrixCV& nabla_smooth_cost_j ) 
{
	Timer::begin("Smoothcost Derivative");
	Vec3d Xi1, Xi2; 
	li->getEndPoints( Xi1, Xi2 ); 

	Vec3d Xj1, Xj2; 
	lj->getEndPoints( Xj1, Xj2 ); 

	// Compute the derivatives of the end points over 
	// the 12 parameters
	static const int indecesXi1[][2] = { {0, 0}, {1, 1}, {2, 2} };
	static const int indecesXi2[][2] = { {0, 3}, {1, 4}, {2, 5} };
	static const int indecesXj1[][2] = { {0, 6}, {1, 7}, {2, 8} };
	static const int indecesXj2[][2] = { {0, 9}, {1,10}, {2,11} };
	static const double values[] = { 1.0, 1.0, 1.0 };
	static const SparseMatrixCV nablaXi1(3, 12, indecesXi1, values, 3 );
	static const SparseMatrixCV nablaXi2(3, 12, indecesXi2, values, 3 );
	static const SparseMatrixCV nablaXj1(3, 12, indecesXj1, values, 3 );
	static const SparseMatrixCV nablaXj2(3, 12, indecesXj2, values, 3 );
	
	Vec3d Pi, Pj, Pi_prime, Pj_prime;
	SparseMatrixCV nablaPi, nablaPj, nablaPi_prime, nablaPj_prime; 

	// TODO be optimized 
	projection_jacobians( Xi1, Xi2, nablaXi1, nablaXi2, tildePi, SparseMatrixCV(3, 12), Pi,       nablaPi );
	projection_jacobians( Xj1, Xj2, nablaXj1, nablaXj2, tildePj, SparseMatrixCV(3, 12), Pj,       nablaPj );
	
	projection_jacobians( Xj1, Xj2, nablaXj1, nablaXj2, Pi,      nablaPi,             Pi_prime, nablaPi_prime );
	projection_jacobians( Xi1, Xi2, nablaXi1, nablaXi2, Pj,      nablaPj,             Pj_prime, nablaPj_prime );

	const double dist_pi_pj2       = max(1e-27, double( (Pi-Pj).dot(Pi-Pj) ) ); 
	const double dist_pi_pi_prime2 = max(1e-27, double( (Pi-Pi_prime).dot(Pi-Pi_prime) ) ); 
	const double dist_pj_pj_prime2 = max(1e-27, double( (Pj-Pj_prime).dot(Pj-Pj_prime))  ); 
	const double dist_pi_pj        = sqrt( dist_pi_pj2 );
	const double dist_pi_pi_prime  = sqrt( dist_pi_pi_prime2 ); 
	const double dist_pj_pj_prime  = sqrt( dist_pj_pj_prime2 ); 

	const SparseMatrixCV nabla_pi_pi_prime = ( Pi - Pi_prime ).t() * ( nablaPi - nablaPi_prime ) / dist_pi_pi_prime;
	const SparseMatrixCV nabla_pj_pj_prime = ( Pj - Pj_prime ).t() * ( nablaPj - nablaPj_prime ) / dist_pj_pj_prime; 
	const SparseMatrixCV nabla_pi_pj       = ( Pi - Pj ).t() * ( nablaPi - nablaPj ) / dist_pi_pj;  
	
	// output result 
	nabla_smooth_cost_i = ( nabla_pi_pi_prime * dist_pi_pj - nabla_pi_pj * dist_pi_pi_prime ) * (1.0 / dist_pi_pj2 * PAIRWISESMOOTH); 
	nabla_smooth_cost_j = ( nabla_pj_pj_prime * dist_pi_pj - nabla_pi_pj * dist_pj_pj_prime ) * (1.0 / dist_pi_pj2 * PAIRWISESMOOTH); 

	Timer::end("Smoothcost Derivative");
}
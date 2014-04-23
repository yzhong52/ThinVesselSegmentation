#include "LevenburgMaquart.h"

#include <iostream> 
#include <iomanip>
#include <Windows.h>
#include <limits>
using namespace std; 

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

inline double compute_energy_datacost_for_one( 
	const Line3D* line_i,
	const Vec3f& pi )
{
	return LOGLIKELIHOOD * LOGLIKELIHOOD * line_i->loglikelihood( pi );
}

double compute_energy_datacost( 
	const vector<Vec3i>& dataPoints,
	const vector<int>& labelings, 
	const vector<Line3D*>& lines )
{
	double energy = 0; 
	for( int site = 0; site < (int) dataPoints.size(); site++ ) {
		int label = labelings[site];
		energy += compute_energy_datacost_for_one( lines[label], dataPoints[site] ); 
	}
	return energy; 
}


void compute_energy_smoothcost_for_pair( 
	const Line3D* line_i,
	const Line3D* line_j,
	const Vec3f& pi_tilde, 
	const Vec3f& pj_tilde,
	double& smooth_cost_i, 
	double& smooth_cost_j )
{
	// single projection
	Vec3f pi = line_i->projection( pi_tilde ); 
	Vec3f pj = line_j->projection( pj_tilde ); 

	// double projection
	Vec3f pi_prime = line_j->projection( pi ); 
	Vec3f pj_prime = line_i->projection( pj ); 

	// distance vector
	Vec3f pi_pj       = pi - pj; 
	Vec3f pi_pi_prime = pi - pi_prime; 
	Vec3f pj_pj_prime = pj - pj_prime;

	// distance
	double dist_pi_pj  = pi_pj.dot(pi_pj); 
	double dist_pi_pi_prime = pi_pi_prime.dot(pi_pi_prime); 
	double dist_pj_pj_prime = pj_pj_prime.dot(pj_pj_prime); 

	if( dist_pi_pj < 1e-20 ) dist_pi_pj = 1e-20; 
	
	if( dist_pi_pi_prime < 1e-10 ) smooth_cost_i = 0; 
	else smooth_cost_i = PAIRWISESMOOTH * PAIRWISESMOOTH * dist_pi_pi_prime / dist_pi_pj; 
	
	if( dist_pj_pj_prime < 1e-10 ) smooth_cost_j = 0; 
	else smooth_cost_j = PAIRWISESMOOTH * PAIRWISESMOOTH * dist_pj_pj_prime / dist_pi_pj; 
}

double compute_energy_smoothcost( 
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

			int l1 = labelings[site];
			int l2 = labelings[site2];

			if( l1==l2 ) continue; 

			double energy_smoothness_i = 0, energy_smoothness_j = 0; 
			compute_energy_smoothcost_for_pair( 
				lines[l1], lines[l2], 
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
	double datacost   = compute_energy_datacost( dataPoints, labelings, lines ); 
	double smoothcost = compute_energy_smoothcost( dataPoints, labelings, lines, indeces ); 
	return datacost + smoothcost; 
}

Mat compute_energy_matrix_datacost( 
	const vector<Vec3i>& dataPoints,
	const vector<int>& labelings, 
	const vector<Line3D*>& lines )
{
	Mat eng( (int) dataPoints.size(), 1, CV_64F ); 
	for( int site = 0; site < (int) dataPoints.size(); site++ ) {
		int label = labelings[site];
		eng.at<double>( site, 0 ) = sqrt( compute_energy_datacost_for_one( lines[label], dataPoints[site] ) ); 
	}
	return eng; 
}


void  projection_jacobians( 
	const Vec3f X1, const Vec3f X2,                                    // two end points of a line
	const SparseMatrixCV& nablaX1, const SparseMatrixCV& nablaX2,          // Jacobians of the end points of the line
	const Vec3f& tildeP,           const SparseMatrixCV& nablaTildeP,      // a point, and the Jacobian of the point
	Vec3f& P, SparseMatrixCV& nablaP )
{	
	Timmer::begin("Projection Jacobians");
	// Assume that: P = T * X1 + (1-T) * X2
	Vec3f X1_X2 = X1 - X2;
	const double A = ( tildeP - X2 ).dot( X1_X2 );
	const double B = ( X1_X2 ).dot( X1_X2 );
	const double T = A / B;

	// Compute the Jacobian matrix for Ai, Bi and Aj, Bj
	//const SparseMatrix MX1_MX2 = MX1 - MX2; 
	const SparseMatrixCV nablaX1_nablaX2 = nablaX1 - nablaX2; 

	//const SparseMatrix nablaA = ( MX1_MX2 ).transpose_multiply( nablaTildeP - nablaX2 ) + ( MTildeP - MX2 ).transpose_multiply( nablaX1_nablaX2 );
	//const SparseMatrix nablaB = ( MX1_MX2 ).transpose_multiply( nablaX1_nablaX2 ) * 2; 
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
	nablaP = MX1 * nablaT + nablaX1 * T + nablaX2 * (1-T) - MX2 * nablaT;

	// This following line is even slower than the above three
	// nablaP = multiply( X1, nablaT ) + nablaX1 * T + nablaX2 * (1-T) - multiply( X2, nablaT );
	Timmer::end("Projection Jacobians");
	
}


SparseMatrixCV compute_datacost_derivative_analytically( const Line3D* l,  const Vec3f tildeP ) {
	Timmer::begin("Datacost Derivative");
	Vec3f X1, X2; 
	l->getEndPoints( X1, X2 );

	static const int indecesM1[][2] = { {0, 0}, {1, 1}, {2, 2} }; 
	static const int indecesM2[][2] = { {0, 3}, {1, 4}, {2, 5} }; 
	static const double values[] = { 1.0, 1.0, 1.0 }; 
	static const SparseMatrixCV nablaX1( 3, 6, indecesM1, values, 3 );
	static const SparseMatrixCV nablaX2( 3, 6, indecesM2, values, 3 );
	
	Vec3f P; 
	SparseMatrixCV nablaP; 
	projection_jacobians( X1, X2, nablaX1, nablaX2, tildeP, SparseMatrixCV(3, 6), P, nablaP );
	
	Vec3d tildeP_P = Vec3d(tildeP) - Vec3d(P); 
	double tildaP_P_lenght = max( 1e-10, sqrt( tildeP_P.dot(tildeP_P) ) ); // l->distanceToLine( tildeP );

	SparseMatrixCV res = tildeP_P.t() * nablaP * ( -1.0 / tildaP_P_lenght * LOGLIKELIHOOD ); 
	Timmer::end("Datacost Derivative");
	return res; 
}


void compute_smoothcost_derivative_analitically(  Line3D* li,   Line3D* lj, const Vec3f& tildePi, const Vec3f& tildePj, SparseMatrixCV& J1, SparseMatrixCV& J2 ) {
	Timmer::begin("Smoothcost Derivative");
	Vec3f Xi1, Xi2; 
	li->getEndPoints( Xi1, Xi2 ); 

	Vec3f Xj1, Xj2; 
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
	
	Vec3f Pi, Pj, Pi_prime, Pj_prime;
	SparseMatrixCV nablaPi, nablaPj, nablaPi_prime, nablaPj_prime; 

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

	// Convert the end points into matrix form
	const SparseMatrixCV MPi( Pi );
	const SparseMatrixCV MPj( Pj ); 
	const SparseMatrixCV MPi_prime( Pi_prime ); 
	const SparseMatrixCV MPj_prime( Pj_prime ); 
	
	SparseMatrixCV nabla_pi_pi_prime = ( MPi - MPi_prime ).t() * ( nablaPi - nablaPi_prime ) ; 
	nabla_pi_pi_prime = nabla_pi_pi_prime / dist_pi_pi_prime;

	SparseMatrixCV nabla_pj_pj_prime = ( ( MPj - MPj_prime ).t() * ( nablaPj - nablaPj_prime ) ) / dist_pj_pj_prime; 
	SparseMatrixCV nabla_pi_pj       = ( MPi - MPj ).t() * ( nablaPi - nablaPj ) / dist_pi_pj; 

	J1 = nabla_pi_pi_prime * dist_pi_pj - nabla_pi_pj * dist_pi_pi_prime; 
	J2 = nabla_pj_pj_prime * dist_pi_pj - nabla_pi_pj * dist_pj_pj_prime; 
	Timmer::end("Smoothcost Derivative");
}

void LevenburgMaquart::reestimate(const vector<Vec3i>& dataPoints,
	const vector<int>& labelings, 
	const vector<Line3D*>& lines, 
	const Data3D<int>& indeces )
{
	double lambda = 1e2; 

	// Data for Jacobian matrix
	//  - # of cols: number of data points; 
	//  - # of rows: number of parameters for all the line models
	vector<double> Jacobian_nzv;
	vector<int>    Jacobian_colindx;
	vector<int>    Jacobian_rowptr(1, 0);

	int numOfParametersPerLine = lines[0]->getNumOfParameters(); 

	for( int lmiter = 0; lambda < 10e50 && lambda > 10e-100 && lmiter<30; lmiter++ ) { 
		Mat_<double> energy_matrix = Mat_<double>( 0, 1 );

		// // // // // // // // // // // // // // // // // // 
		// Construct Jacobian Matrix - for data cost
		// // // // // // // // // // // // // // // // // // 

		energy_matrix = compute_energy_matrix_datacost( dataPoints, labelings, lines ); 
		
		for( int site=0; site < dataPoints.size(); site++ ) {
			int label = labelings[site]; 
			// Computing Derivative Analytically
			SparseMatrixCV matrix = compute_datacost_derivative_analytically( lines[label], dataPoints[site] ); 
			
			int N;
			const double* non_zero_value = NULL;
			const int * column_index = NULL;
			const int* row_pointer = NULL; 
			matrix.getRowMatrixData( N, &non_zero_value, &column_index, &row_pointer ); 
			
			assert( matrix.row()==1 && N==numOfParametersPerLine && "Number of row is not correct for Jacobian matrix" );
			for( int i=0; i<N; i++ ) {
				Jacobian_nzv.push_back( non_zero_value[i] );
				Jacobian_colindx.push_back( column_index[i] + site * N ); 
			}
			Jacobian_rowptr.push_back( (int) Jacobian_nzv.size() ); 
		}


		// // // // // // // // // // // // // // // // // // 
		// Construct Jacobian Matrix - for smooth cost
		// // // // // // // // // // // // // // // // // // 
		for( int site = 0; site < dataPoints.size(); site++ ) { // For each data point
			for( int neibourIndex=0; neibourIndex<13; neibourIndex++ ) { // find it's neighbour
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

				// Mat JJ = Mat::zeros( 2, numOfParametersTotal, CV_64F ); 

				int l1 = labelings[site];
				int l2 = labelings[site2];
				
				if( l1==l2 ) continue; // TODO

				double smoothcost_i_before = 0, smoothcost_j_before = 0;
				compute_energy_smoothcost_for_pair( 
					lines[l1], lines[l2], 
					dataPoints[site], dataPoints[site2], 
					smoothcost_i_before, smoothcost_j_before ); 
				
				// add more rows to energy_matrix according to smooth cost 
				energy_matrix.push_back( sqrt( smoothcost_i_before ) ); 
				energy_matrix.push_back( sqrt( smoothcost_j_before ) ); 
				
				////// Computing derivative of pair-wise smooth cost analytically
				SparseMatrixCV J[2];
				compute_smoothcost_derivative_analitically(  lines[l1], lines[l2], 
					dataPoints[site], dataPoints[site2], J[0], J[1] ); 

				for( int ji = 0; ji<2; ji++ ) {
					int N;
					const double* non_zero_value = NULL;
					const int * column_index = NULL;
					const int* row_pointer = NULL; 
					J[ji].getRowMatrixData( N, &non_zero_value, &column_index, &row_pointer ); 
					assert( J[ji].row()==1 && N==2*numOfParametersPerLine && "Number of row is not correct for Jacobian matrix" );

					const int N2 = N/2; 
					for( int i=0; i<N2; i++ ) {
						Jacobian_nzv.push_back( non_zero_value[i] );
						Jacobian_colindx.push_back( column_index[i] + site * N2 ); 
					}
					for( int i=0; i<N2; i++ ) {
						Jacobian_nzv.push_back( non_zero_value[i + N2] );
						Jacobian_colindx.push_back( column_index[i + N2] + site2 * N2 ); 
					}
					Jacobian_rowptr.push_back( (int) Jacobian_nzv.size() ); 
				}
			} // for each pair of pi and pj
		} // end of contruction of Jacobian Matrix


		SparseMatrixCV Jacobian = SparseMatrix(
			(int) Jacobian_rowptr.size(), 
			(int) lines.size() * numOfParametersPerLine, 
			Jacobian_nzv, Jacobian_colindx, Jacobian_rowptr );

		SparseMatrixCV A = Jacobian.t() * Jacobian + SparseMatrixCV::I( Jacobian.row() );

		Mat_<double> B = Jacobian.t() * energy_matrix; 

		Mat_<double> X;
		solve( A, B, X );
		
		X = -X; 
		
		//for( int i=0; i<X.rows; i++ ) {
		//	std::cout << std::setw(14) << std::scientific << X.at<double>(i) << "  ";
		//}
		//cout << endl;
		//Sleep(500);
		


		double energy_before = compute_energy( dataPoints, labelings, lines, indeces );

		for( int label=0; label < lines.size(); label++ ) {
			for( int i=0; i < numOfParametersPerLine; i++ ) {
				const double& delta = X.at<double>( label * numOfParametersPerLine + i ); 
				lines[label]->updateParameterWithDelta( i, delta ); 
			}
		}

		double new_energy = compute_energy( dataPoints, labelings, lines, indeces );

		// the smaller lambda is, the faster it converges
		// the bigger lambda is, the slower it converges
		if( new_energy <= energy_before ) { // if energy is decreasing 
			cout << "- ";
			energy_before = new_energy; 
			lambda *= 0.71; 
		} else {
			cout << "+ ";
			for( int label=0; label < lines.size(); label++ ) {
				for( int i=0; i < numOfParametersPerLine; i++ ) {
					const double& delta = X.at<double>( label * numOfParametersPerLine + i ); 
					lines[label]->updateParameterWithDelta( i, -delta ); 
				}
			}
			lambda *= 2.13; 
		}

		// TODO: this might be time consuming
		// update the end points of the line
		vector<float> minT( (int) labelings.size(), (std::numeric_limits<float>::max)() );
		vector<float> maxT( (int) labelings.size(), (std::numeric_limits<float>::min)() );
		for( int site = 0; site < dataPoints.size(); site++ ) { // For each data point
			int label = labelings[site]; 
			Vec3f p1, p2;
			lines[label]->getEndPoints( p1, p2 ); 

			Vec3f& pos = p1;
			Vec3f dir = p2 - p1; 
			dir /= sqrt( dir.dot( dir ) ); // normalize the direction 
			float t = ( Vec3f(dataPoints[site]) - pos ).dot( dir );
			maxT[label] = max( t+1, maxT[label] );
			minT[label] = min( t-1, minT[label] );
		}
		for( int label=0; label<labelings.size(); label++ ) {
			if( minT[label] < maxT[label] ) {
				Vec3f p1, p2;
				lines[label]->getEndPoints( p1, p2 ); 

				Vec3f& pos = p1;
				Vec3f dir = p2 - p1; 
				dir /= sqrt( dir.dot( dir ) ); // normalize the direction 
				
				lines[label]->setPositions( pos + dir * minT[label], pos + dir * maxT[label] ); 
			}
		}
	}
}
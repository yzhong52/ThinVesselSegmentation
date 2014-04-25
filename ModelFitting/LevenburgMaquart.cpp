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
#include "ModelSet.h"


#include "../SparseMatrixCV/SparseMatrixCV.h"
#ifdef _DEBUG
	#pragma comment(lib,"../x64/Debug/SparseMatrixCV.lib")
	#pragma comment(lib,"../x64/Debug/SparseMatrix.lib")
#else
	#pragma comment(lib,"../x64/Release/SparseMatrixCV.lib")
	#pragma comment(lib,"../x64/Release/SparseMatrix.lib")
#endif

#include "EnergyFunctions.h"

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

	nablaP = MX1 * nablaT + nablaX1 * T + nablaX2 * (1-T) - MX2 * nablaT;

	Timer::end("Projection Jacobians");
	
}


SparseMatrixCV compute_datacost_derivative_analytically( const Line3D* l,  const Vec3d tildeP ) {
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
	
	Vec3d tildeP_P = Vec3d(tildeP) - Vec3d(P); 
	double tildaP_P_lenght = max( 1e-20, sqrt( tildeP_P.dot(tildeP_P) ) ); // l->distanceToLine( tildeP );

	SparseMatrixCV res = ( -1.0 / tildaP_P_lenght * LOGLIKELIHOOD ) * tildeP_P.t() * nablaP ; 

	Timer::end("Datacost Derivative");
	return res; 
}


void compute_smoothcost_derivative_analitically(  Line3D* li,   Line3D* lj, const Vec3d& tildePi, const Vec3d& tildePj,
	SparseMatrixCV& nabla_smooth_cost_i, SparseMatrixCV& nabla_smooth_cost_j ) 
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
	
	const SparseMatrixCV nabla_pi_pi_prime = ( MPi - MPi_prime ).t() * ( nablaPi - nablaPi_prime ) / dist_pi_pi_prime;

	const SparseMatrixCV nabla_pj_pj_prime = ( ( MPj - MPj_prime ).t() * ( nablaPj - nablaPj_prime ) ) / dist_pj_pj_prime; 
	const SparseMatrixCV nabla_pi_pj       = ( MPi - MPj ).t() * ( nablaPi - nablaPj ) / dist_pi_pj;  
	
	nabla_smooth_cost_i = ( nabla_pi_pi_prime * dist_pi_pj - nabla_pi_pj * dist_pi_pi_prime ) * (1.0 / dist_pi_pj2 * PAIRWISESMOOTH); 
	nabla_smooth_cost_j = ( nabla_pj_pj_prime * dist_pi_pj - nabla_pi_pj * dist_pj_pj_prime ) * (1.0 / dist_pi_pj2 * PAIRWISESMOOTH); 

	Timer::end("Smoothcost Derivative");
}

void LevenburgMaquart::reestimate(const vector<Vec3i>& dataPoints,
	const vector<int>& labelings, 
	const ModelSet<Line3D>& modelset, 
	const Data3D<int>& indeces )
{
	const vector<Line3D*>& lines = modelset.models; 

	int numParamPerLine = lines[0]->getNumOfParameters(); 
	
	double energy_before = compute_energy( dataPoints, labelings, lines, indeces );

	double lambda = 1e2;
	int lmiter = 0; 
	for( lambda = 1e2, lmiter = 0; lambda < 10e50 && lambda > 10e-100 && lmiter<230; lmiter++ ) { 

		// Data for Jacobian matrix
		//  - # of cols: number of data points; 
		//  - # of rows: number of parameters for all the line models
		vector<double> Jacobian_nzv;
		vector<int>    Jacobian_colindx;
		vector<int>    Jacobian_rowptr(1, 0);

		Mat_<double> energy_matrix = Mat_<double>( 0, 1 );

		// // // // // // // // // // // // // // // // // // 
		// Construct Jacobian Matrix -  data cost
		// // // // // // // // // // // // // // // // // // 

		energy_matrix = compute_matrix_datacost( dataPoints, labelings, lines ); 
		
		for( int site=0; site < dataPoints.size(); site++ ) {
			int label = labelings[site]; 

			// Computing Derivative Analytically
			SparseMatrixCV matrix = compute_datacost_derivative_analytically( lines[label], dataPoints[site] ); 

			int N;
			const double* non_zero_value = NULL;
			const int * column_index = NULL;
			const int* row_pointer = NULL; 
			matrix.getRowMatrixData( N, &non_zero_value, &column_index, &row_pointer ); 
			assert( matrix.row()==1 && matrix.col()==numParamPerLine && "Number of row is not correct for Jacobian matrix" );
			for( int i=0; i<N; i++ ) {
				Jacobian_nzv.push_back( non_zero_value[i] );
				Jacobian_colindx.push_back( column_index[i] + site * N ); 
			}
			Jacobian_rowptr.push_back( (int) Jacobian_nzv.size() ); 
			
		}

		
		// // // // // // // // // // // // // // // // // // 
		// Construct Jacobian Matrix - smooth cost 
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

				int l1 = labelings[site];
				int l2 = labelings[site2];
				
				if( l1==l2 ) continue; // TODO

				double smoothcost_i_before = 0, smoothcost_j_before = 0;
				compute_energy_smoothcost_for_pair( 
					lines[l1], lines[l2], dataPoints[site], dataPoints[site2], 
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
					assert( J[ji].row()==1 && J[ji].col()==2*numParamPerLine && "Number of row is not correct for Jacobian matrix" );

					int n1; 
					for( n1=0; n1<N && column_index[n1] < numParamPerLine; n1++ ) {
						Jacobian_nzv.push_back( non_zero_value[n1] );
						Jacobian_colindx.push_back( column_index[n1] + site * numParamPerLine ); 
					}
					int n2 = n1; 
					for( ; n2<N; n2++ ) {
						Jacobian_nzv.push_back( non_zero_value[n2] );
						Jacobian_colindx.push_back( column_index[n2] + (site2-1) * numParamPerLine ); 
					}
					Jacobian_rowptr.push_back( (int) Jacobian_nzv.size() ); 
				}

			} // for each pair of pi and pj
		} // end of contruction of Jacobian Matrix

		SparseMatrixCV Jacobian = SparseMatrix(
			(int) Jacobian_rowptr.size() - 1, 
			(int) lines.size() * numParamPerLine, 
			Jacobian_nzv, Jacobian_colindx, Jacobian_rowptr );

		SparseMatrixCV A = Jacobian.t() * Jacobian + SparseMatrixCV::I( Jacobian.col() ) * lambda;
		Mat_<double> B = Jacobian.t() * energy_matrix; 
		Mat_<double> X;

		Timer::begin( "Linear Solver" ); 
		solve( A, B, X );
		X = -X; 
		Timer::end( "Linear Solver" ); 
		
		//for( int i=0; i<X.rows; i++ ) {
		//	std::cout << std::setw(14) << std::scientific << X.at<double>(i) << "  ";
		//}
		//cout << endl;
		//Sleep(500);
		
		for( int label=0; label < lines.size(); label++ ) {
			for( int i=0; i < numParamPerLine; i++ ) {
				const double& delta = X.at<double>( label * numParamPerLine + i ); 
				lines[label]->updateParameterWithDelta( i, delta ); 
			}
		}

		double new_energy = compute_energy( dataPoints, labelings, lines, indeces );

		// the smaller lambda is, the faster it converges
		// the bigger lambda is, the slower it converges
		static int energy_increase_count = 0; 
		if( new_energy < energy_before ) { // if energy is decreasing 
			cout << "- ";
			energy_before = new_energy; 
			lambda *= 0.71; 
			energy_increase_count = 0; 
		} else {
			
			cout << "+ ";
			for( int label=0; label < lines.size(); label++ ) {
				for( int i=0; i < numParamPerLine; i++ ) {
					const double& delta = X.at<double>( label * numParamPerLine + i ); 
					lines[label]->updateParameterWithDelta( i, -delta ); 
				}
			}
			lambda *= 1.72; 
			
			// If energy_increase_count in three consecutive iterations
			// then the nenergy is probabaly converged
			if( ++energy_increase_count>=1 ) break; 
		}

		// TODO: this might be time consuming
		// update the end points of the line
		Timer::begin( "Update End Points" ); 
		vector<double> minT( (int) labelings.size(), (std::numeric_limits<double>::max)() );
		vector<double> maxT( (int) labelings.size(), (std::numeric_limits<double>::min)() );
		for( int site = 0; site < dataPoints.size(); site++ ) { // For each data point
			int label = labelings[site]; 
			Vec3d p1, p2;
			lines[label]->getEndPoints( p1, p2 ); 

			Vec3d& pos = p1;
			Vec3d dir = p2 - p1; 
			dir /= sqrt( dir.dot( dir ) ); // normalize the direction 
			double t = ( Vec3d(dataPoints[site]) - pos ).dot( dir );
			maxT[label] = max( t+1, maxT[label] );
			minT[label] = min( t-1, minT[label] );
		}
		for( int label=0; label<labelings.size(); label++ ) {
			if( minT[label] < maxT[label] ) {
				Vec3d p1, p2;
				lines[label]->getEndPoints( p1, p2 ); 

				Vec3d& pos = p1;
				Vec3d dir = p2 - p1; 
				dir /= sqrt( dir.dot( dir ) ); // normalize the direction 
				
				lines[label]->setPositions( pos + dir * minT[label], pos + dir * maxT[label] ); 
			}
		}
		Timer::end( "Update End Points" ); 

		//cout << endl;
		//for( int i=2; i>=0; i-- ){
		//	cout << '\r' << "Serializing models in " << i << " seconds... "; Sleep( 100 ); 
		//}
		//modelset.serialize( "output/Line3DTwoPoint.model" ); 
		//cout << "Serialization done. " << endl;
	}

}
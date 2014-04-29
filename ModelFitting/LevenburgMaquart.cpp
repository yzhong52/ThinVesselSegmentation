#include "LevenburgMaquart.h"
#include <iostream> 
#include <iomanip>
#include <Windows.h>
#include <limits>
#include "Line3D.h" 
#include "Neighbour26.h"
#include "Data3D.h"
#include "Timer.h"
#include "ModelSet.h"
#include "EnergyFunctions.h"
#include "SparseMatrixCV\SparseMatrixCV.h" 
#include <omp.h>


using namespace std; 
using namespace cv;



// X1, X2: 3 * 1, two end points of the line
// nablaX: 3 * 12, 3 non-zero values
// nablaP: 3 * 12
void  LevenburgMaquart::Jacobian_projection( 
	const Vec3d& X1, const Vec3d& X2,                                    // two end points of a line
	const SparseMatrixCV& nablaX1, const SparseMatrixCV& nablaX2,          // Jacobians of the end points of the line
	const Vec3d& tildeP,           const SparseMatrixCV& nablaTildeP,      // a point, and the Jacobian of the point
	Vec3d& P, SparseMatrixCV& nablaP )
{	
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
}


SparseMatrixCV LevenburgMaquart::Jacobian_datacost_for_one( const Line3D* l,  const Vec3d tildeP, int site ) {
	Vec3d X1, X2; 
	l->getEndPoints( X1, X2 );

	static const int indecesM1[][2] = { {0, 0}, {1, 1}, {2, 2} }; 
	static const int indecesM2[][2] = { {0, 3}, {1, 4}, {2, 5} }; 
	static const double values[] = { 1.0, 1.0, 1.0 }; 
	static const SparseMatrixCV nablaX1( 3, 6, indecesM1, values, 3 );
	static const SparseMatrixCV nablaX2( 3, 6, indecesM2, values, 3 );
	
	Jacobian_projection( X1, X2, nablaX1, nablaX2, tildeP, SparseMatrixCV(3, 6), 
		P[site],
		nablaP[site] );
	
	const Vec3d tildeP_P = Vec3d(tildeP) - P[site]; 
	const double tildaP_P_lenght = max( 1e-20, sqrt( tildeP_P.dot(tildeP_P) ) ); // l->distanceToLine( tildeP );

	const SparseMatrixCV res = ( -1.0/tildaP_P_lenght*LOGLIKELIHOOD ) * tildeP_P.t() * nablaP[site]; 

	return res; 
}


void LevenburgMaquart::Jacobian_smoothcost_for_pair( const Line3D* li, const Line3D* lj, 
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
	Jacobian_projection( Xi1, Xi2, nablaXi1, nablaXi2, tildePi, SparseMatrixCV(3, 12), Pi,       nablaPi );
	Jacobian_projection( Xj1, Xj2, nablaXj1, nablaXj2, tildePj, SparseMatrixCV(3, 12), Pj,       nablaPj );
	
	Jacobian_projection( Xj1, Xj2, nablaXj1, nablaXj2, Pi,      nablaPi,             Pi_prime, nablaPi_prime );
	Jacobian_projection( Xi1, Xi2, nablaXi1, nablaXi2, Pj,      nablaPj,             Pj_prime, nablaPj_prime );

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

void LevenburgMaquart::Jacobian_datacost(
	vector<double>& Jacobian_nzv, 
	vector<int>&    Jacobian_colindx, 
	vector<int>&    Jacobian_rowptr,
	vector<double>& energy_matrix )
{
	const vector<Line3D*>& lines = modelset.models; 
	int numParamPerLine = lines[0]->getNumOfParameters(); 

	for( int site=0; site < dataPoints.size(); site++ ) {
		const int& label = labelings[site]; 

		// computing datacost 
		const double datacost_i = compute_datacost_for_one( lines[label], dataPoints[site] ); 

		// Computing derivative for data cost analytically
		SparseMatrixCV J_datacost = Jacobian_datacost_for_one( 
			lines[label], dataPoints[site], site ); 

		energy_matrix.push_back( sqrt(datacost_i) ); 

		int N;
		const double* non_zero_value = NULL;
		const int * column_index = NULL;
		const int* row_pointer = NULL; 
		J_datacost.getRowMatrixData( N, &non_zero_value, &column_index, &row_pointer ); 
		assert( J_datacost.row()==1 && J_datacost.col()==numParamPerLine && "Number of row is not correct for Jacobian matrix" );
		for( int i=0; i<N; i++ ) {
			Jacobian_nzv.push_back( non_zero_value[i] );
			Jacobian_colindx.push_back( column_index[i] + site * N ); 
		}
		Jacobian_rowptr.push_back( (int) Jacobian_nzv.size() ); 
	}
}


void LevenburgMaquart::Jacobian_smoothcost(
	vector<double>& Jacobian_nzv, 
	vector<int>&    Jacobian_colindx, 
	vector<int>&    Jacobian_rowptr,
	vector<double>& energy_matrix )
{
	const vector<Line3D*>& lines = modelset.models; 
	int numParamPerLine = lines[0]->getNumOfParameters(); 
	const Data3D<int>& indeces = labelIDs; 

	for( int site = 0; site < dataPoints.size(); site++ ) { // For each data point
		for( int neibourIndex=0; neibourIndex<13; neibourIndex++ ) { // find it's neighbour
			// the neighbour position
			int x, y, z; 
			Neighbour26::getNeigbour( neibourIndex, 
				dataPoints[site][0], 
				dataPoints[site][1], 
				dataPoints[site][2], 
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
			compute_smoothcost_for_pair( lines[l1], lines[l2], 
				dataPoints[site], dataPoints[site2], 
				smoothcost_i_before, smoothcost_j_before ); 

			// add more rows to energy_matrix according to smooth cost 
			energy_matrix.push_back( sqrt( smoothcost_i_before ) ); 
			energy_matrix.push_back( sqrt( smoothcost_j_before ) ); 

			////// Computing derivative of pair-wise smooth cost analytically
			SparseMatrixCV J[2];
			Jacobian_smoothcost_for_pair(  lines[l1], lines[l2], 
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
}



void LevenburgMaquart::Jacobian_smoothcost_openmp(
	vector<double>& Jacobian_nzv, 
	vector<int>&    Jacobian_colindx, 
	vector<int>&    Jacobian_rowptr,
	vector<double>&	energy_matrix )
{
	// // // // // // // // // // // // // // // // // // 
	// Construct Jacobian Matrix - smooth cost 
	// // // // // // // // // // // // // // // // // // 
	const vector<Line3D*>& lines = modelset.models; 
	int numParamPerLine = lines[0]->getNumOfParameters(); 
	const Data3D<int>& indeces = labelIDs; 

	int max_num_threads = omp_get_max_threads(); // maximum number of thread
	vector<unsigned int> nzv_size( max_num_threads, 0); 
	vector<unsigned int> nzv_rows( max_num_threads, 0); 
	vector<unsigned int> accumulate_nzv_size( max_num_threads, 0); 
	vector<unsigned int> accumulate_nzv_rows( max_num_threads, 0); 
#pragma omp parallel /* Fork a team of threads*/ 
	{
		// local variables for different processes
		vector<double> Jacobian_nzv_loc;
		vector<int>    Jacobian_colindx_loc;
		vector<int>    Jacobian_rowptr_loc;
		vector<double>   energy_matrix_loc;
#pragma omp for
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
				compute_smoothcost_for_pair( 
					lines[l1], lines[l2], dataPoints[site], dataPoints[site2], 
					smoothcost_i_before, smoothcost_j_before ); 

				// add more rows to energy_matrix according to smooth cost 
				energy_matrix_loc.push_back( sqrt( smoothcost_i_before ) ); 
				energy_matrix_loc.push_back( sqrt( smoothcost_j_before ) ); 

				////// Computing derivative of pair-wise smooth cost analytically
				SparseMatrixCV J[2];
				Jacobian_smoothcost_for_pair(  lines[l1], lines[l2], 
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
						Jacobian_nzv_loc.push_back( non_zero_value[n1] );
						Jacobian_colindx_loc.push_back( column_index[n1] + site * numParamPerLine ); 
					}
					int n2 = n1; 
					for( ; n2<N; n2++ ) {
						Jacobian_nzv_loc.push_back( non_zero_value[n2] );
						Jacobian_colindx_loc.push_back( column_index[n2] + (site2-1) * numParamPerLine ); 
					}
					Jacobian_rowptr_loc.push_back( (int) Jacobian_nzv_loc.size() ); 
				}

			} // for each pair of pi and pj
		} // end of contruction of Jacobian Matrix

		// obtatin current thread id
		int tid = omp_get_thread_num(); 
		// copy the current number size to global vectors nzv_size and nzv
		// 'global' here is with respect to 'non-local' for threads
		nzv_size[tid] = (unsigned int) Jacobian_nzv_loc.size(); 
		nzv_rows[tid] = (unsigned int) Jacobian_rowptr_loc.size(); 
		// original size of the global vectors
		unsigned int old_nzv_size = (unsigned int) Jacobian_nzv.size();
		unsigned int old_num_rows = (unsigned int) energy_matrix.size(); 

		smart_assert( Jacobian_nzv.size()==Jacobian_colindx.size(), "vector size mismatch. " ); 
		smart_assert( Jacobian_rowptr.size()-1==energy_matrix.size(), "vector size mismatch. " ); 

#pragma omp barrier // wait for all thread to execute till this point 
		if( tid==0 ) {
			int nThreads = omp_get_num_threads();
			accumulate_nzv_size[0] = nzv_size[0]; 
			accumulate_nzv_rows[0] = nzv_rows[0]; 
			for( int i=1; i<nThreads; i++ ) {
				accumulate_nzv_size[i] = nzv_size[i] + accumulate_nzv_size[i-1]; 
				accumulate_nzv_rows[i] = nzv_rows[i] + accumulate_nzv_rows[i-1]; 
			}
			
			// resize the result vector, make them bigger!
			Jacobian_nzv.resize(     old_nzv_size + accumulate_nzv_size[nThreads-1] ); 
			Jacobian_colindx.resize( old_nzv_size + accumulate_nzv_size[nThreads-1] ); 
			Jacobian_rowptr.resize(  old_num_rows + accumulate_nzv_rows[nThreads-1] + 1); 
			energy_matrix.resize(    old_num_rows + accumulate_nzv_rows[nThreads-1] );
		}

#pragma omp barrier // wait for all thread to execute till this point 

		// copy data to result vector 
		int nzv_offset = old_nzv_size; 
		if(tid>0) nzv_offset += accumulate_nzv_size[tid-1];
		// The following two memcpy is equivalent to the following for loop 
		//for( unsigned int i=0; i<nzv_size[tid]; i++ ) {
		//	Jacobian_nzv[nzv_offset + i] = Jacobian_colindx_loc[i]; 
		//	Jacobian_colindx[nzv_offset+i] = Jacobian_colindx_loc[i];
		//}
		memcpy( &Jacobian_nzv[nzv_offset], &Jacobian_nzv_loc[0], nzv_size[tid] * sizeof(double) ); 
		memcpy( &Jacobian_colindx[nzv_offset], &Jacobian_colindx_loc[0], nzv_size[tid] * sizeof(int) ); 

		int row_offset = old_num_rows; 
		if( tid>0 ) row_offset += accumulate_nzv_rows[tid-1]; 
		int data_offset = old_nzv_size;
		if( tid>0 ) data_offset += accumulate_nzv_size[tid-1]; 
		for( int i=0; i<Jacobian_rowptr_loc.size(); i++ ) {
			Jacobian_rowptr[ i + row_offset + 1 ] = Jacobian_rowptr_loc[i] + data_offset; 
		}

		memcpy( &energy_matrix[row_offset],
			&energy_matrix_loc[0], 
			nzv_rows[tid] * sizeof(double) ); 
	}
}



void LevenburgMaquart::Jacobian_smoothcost_openmp_critical_section(
	vector<double>& Jacobian_nzv, 
	vector<int>&    Jacobian_colindx, 
	vector<int>&    Jacobian_rowptr,
	vector<double>&	energy_matrix )
{
	// // // // // // // // // // // // // // // // // // 
	// Construct Jacobian Matrix - smooth cost 
	// // // // // // // // // // // // // // // // // // 
	const vector<Line3D*>& lines = modelset.models; 
	int numParamPerLine = lines[0]->getNumOfParameters(); 
	const Data3D<int>& indeces = labelIDs; 

	int max_num_threads = omp_get_max_threads(); // maximum number of thread
	vector<unsigned int> nzv_size( max_num_threads, 0); 
	vector<unsigned int> nzv_rows( max_num_threads, 0); 
	vector<unsigned int> accumulate_nzv_size( max_num_threads, 0); 
	vector<unsigned int> accumulate_nzv_rows( max_num_threads, 0); 
#pragma omp parallel /* Fork a team of threads*/ 
	{
		// local variables for different processes
		vector<double> Jacobian_nzv_loc;
		vector<int>    Jacobian_colindx_loc;
		vector<int>    Jacobian_rowptr_loc;
		vector<double>   energy_matrix_loc;
#pragma omp for
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
				compute_smoothcost_for_pair( 
					lines[l1], lines[l2], dataPoints[site], dataPoints[site2], 
					smoothcost_i_before, smoothcost_j_before ); 

				// add more rows to energy_matrix according to smooth cost 
				energy_matrix_loc.push_back( sqrt( smoothcost_i_before ) ); 
				energy_matrix_loc.push_back( sqrt( smoothcost_j_before ) ); 

				////// Computing derivative of pair-wise smooth cost analytically
				SparseMatrixCV J[2];
				Jacobian_smoothcost_for_pair(  lines[l1], lines[l2], 
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
						Jacobian_nzv_loc.push_back( non_zero_value[n1] );
						Jacobian_colindx_loc.push_back( column_index[n1] + site * numParamPerLine ); 
					}
					int n2 = n1; 
					for( ; n2<N; n2++ ) {
						Jacobian_nzv_loc.push_back( non_zero_value[n2] );
						Jacobian_colindx_loc.push_back( column_index[n2] + (site2-1) * numParamPerLine ); 
					}
					Jacobian_rowptr_loc.push_back( (int) Jacobian_nzv_loc.size() ); 
				}

			} // for each pair of pi and pj
		} // end of contruction of Jacobian Matrix


#pragma omp critical 
		{
			Jacobian_nzv.insert( Jacobian_nzv.end(), Jacobian_nzv_loc.begin(), Jacobian_nzv_loc.end());
			Jacobian_colindx.insert( Jacobian_colindx.end(), Jacobian_colindx_loc.begin(), Jacobian_colindx_loc.end());
			int offset = Jacobian_rowptr.back();
			for( int i=0; i<Jacobian_rowptr_loc.size(); i++ ) {
				Jacobian_rowptr.push_back( Jacobian_rowptr_loc[i] + offset ); 
			}
			energy_matrix.insert( energy_matrix.begin(), 
				energy_matrix_loc.begin(), 
				energy_matrix_loc.end() ); 
		}

	}
}

void LevenburgMaquart::reestimate( void )
{
	const vector<Line3D*>& lines = modelset.models; 
	const Data3D<int>& indeces = labelIDs; 

	if( lines.size()==0 ) {
		cout << "No line models available" << endl;
		return; 
	}
	int numParamPerLine = lines[0]->getNumOfParameters(); 

	double energy_before = compute_energy( dataPoints, labelings, lines, indeces );

	
	P = vector<Vec3d>( dataPoints.size() );
	nablaP = vector<SparseMatrixCV>( dataPoints.size() );
	
	double lambda = 1e2; // lamda - damping function for levenburg maquart
	int lmiter = 0; // levenburg maquarit iteration count
	for( lmiter = 0; lmiter<50; lmiter++ ) { 

		// Data for Jacobian matrix
		//  - # of cols: number of data points; 
		//  - # of rows: number of parameters for all the line models
		vector<double> Jacobian_nzv;
		vector<int>    Jacobian_colindx;
		vector<int>    Jacobian_rowptr(1, 0);

		vector<double> energy_matrix;

		// // // // // // // // // // // // // // // // // // 
		// Construct Jacobian Matrix -  data cost
		// // // // // // // // // // // // // // // // // // 
		Jacobian_datacost( Jacobian_nzv, Jacobian_colindx, Jacobian_rowptr, energy_matrix );

		// // // // // // // // // // // // // // // // // // 
		// Construct Jacobian Matrix - smooth cost 
		// // // // // // // // // // // // // // // // // // 
		Jacobian_smoothcost_openmp( Jacobian_nzv, Jacobian_colindx, Jacobian_rowptr, energy_matrix );
		
		// Construct Jacobian matrix
		SparseMatrixCV Jacobian = SparseMatrix(
			(int) Jacobian_rowptr.size() - 1, 
			(int) lines.size() * numParamPerLine, 
			Jacobian_nzv, Jacobian_colindx, Jacobian_rowptr );
		
		SparseMatrixCV A = Jacobian.t() * Jacobian + SparseMatrixCV::I( Jacobian.col() ) * lambda;
		// TODO: the following line need to be optimized
		Mat_<double> B = Jacobian.t() * cv::Mat_<double>( (int) energy_matrix.size(), 1, &energy_matrix.front() ) ; 
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
			cout << "- " << new_energy << endl;
			energy_before = new_energy; 
			lambda *= 0.71; 
			energy_increase_count = 0; 
		} else {
			cout << "+ " << new_energy << endl;
			for( int label=0; label < lines.size(); label++ ) {
				for( int i=0; i < numParamPerLine; i++ ) {
					const double& delta = X.at<double>( label * numParamPerLine + i ); 
					lines[label]->updateParameterWithDelta( i, -delta ); 
				}
			}
			lambda *= 1.72; 

			// If energy_increase_count in three consecutive iterations
			// then the nenergy is probabaly converged
			if( ++energy_increase_count>=2 ) break; 
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

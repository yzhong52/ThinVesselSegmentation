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

LevenburgMaquart::LevenburgMaquart( const vector<Vec3i>& dataPoints, const vector<int>& labelings, 
	const ModelSet<Line3D>& modelset, const Data3D<int>& labelIDs ) 
	: tildaP( dataPoints ), labelID( labelings )
	, modelset( modelset ), labelID3d( labelIDs )
	, lines( modelset.models )
{
	numParamPerLine = lines[0]->getNumOfParameters(); 
	numParam = numParamPerLine * (int) lines.size(); 

	P = vector<Vec3d>( tildaP.size() );              // projection points of original points
	nablaP = vector<SparseMatrixCV>( tildaP.size() ); // Jacobian matrix of the porjeciton points 
}

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


SparseMatrixCV LevenburgMaquart::Jacobian_datacost_for_one( const int& site ) {
	int label = labelID[site];
	const Line3D* line = lines[label]; 

	Vec3d X1, X2; 
	line->getEndPoints( X1, X2 );

	const int indecesM1[][2] = { 
		{0, 0 + label * numParamPerLine}, 
		{1, 1 + label * numParamPerLine}, 
		{2, 2 + label * numParamPerLine}}; 
	const int indecesM2[][2] = { 
		{0, 3 + label * numParamPerLine}, 
		{1, 4 + label * numParamPerLine}, 
		{2, 5 + label * numParamPerLine}}; 
	static const double values[] = { 1.0, 1.0, 1.0 }; 
	const SparseMatrixCV nablaX1( 3, numParam, indecesM1, values, 3 );
	const SparseMatrixCV nablaX2( 3, numParam, indecesM2, values, 3 );
	
	Jacobian_projection( X1, X2, nablaX1, nablaX2, tildaP[site], SparseMatrixCV(3, numParam), 
		P[site],
		nablaP[site] );
	
	const Vec3d tildeP_P = Vec3d(tildaP[site]) - P[site]; 
	const double tildaP_P_lenght = max( 1e-20, sqrt( tildeP_P.dot(tildeP_P) ) ); // l->distanceToLine( tildeP );

	const SparseMatrixCV res = ( -1.0/tildaP_P_lenght*LOGLIKELIHOOD ) * tildeP_P.t() * nablaP[site]; 

	return res; 
}



void LevenburgMaquart::Jacobian_smoothcost_for_pair( 
	const int& sitei, const int& sitej, 
	SparseMatrixCV& nabla_smooth_cost_i,
	SparseMatrixCV& nabla_smooth_cost_j  ) 
{
	int labeli = labelID[sitei];
	int labelj = labelID[sitej];
	const Line3D* linei = lines[labeli]; 
	const Line3D* linej = lines[labelj]; 

	Vec3d Xi1, Xi2; 
	linei->getEndPoints( Xi1, Xi2 ); 
	Vec3d Xj1, Xj2; 
	linej->getEndPoints( Xj1, Xj2 ); 

	const Vec3d tildePi = tildaP[sitei]; 
	const Vec3d tildePj = tildaP[sitej]; 

	// Compute the derivatives of the end points over 
	// the 12 parameters
	const int indecesXi1[][2] = { 
		{0, 0 + labeli * numParamPerLine}, 
		{1, 1 + labeli * numParamPerLine}, 
		{2, 2 + labeli * numParamPerLine}}; 
	const int indecesXi2[][2] = { 
		{0, 3 + labeli * numParamPerLine}, 
		{1, 4 + labeli * numParamPerLine}, 
		{2, 5 + labeli * numParamPerLine}}; 
	const int indecesXj1[][2] = { 
		{0, 0 + labelj * numParamPerLine}, 
		{1, 1 + labelj * numParamPerLine}, 
		{2, 2 + labelj * numParamPerLine}}; 
	const int indecesXj2[][2] = { 
		{0, 3 + labelj * numParamPerLine}, 
		{1, 4 + labelj * numParamPerLine}, 
		{2, 5 + labelj * numParamPerLine}}; 
	
	static const double values[] = { 1.0, 1.0, 1.0 };
	const SparseMatrixCV nablaXi1(3, numParam, indecesXi1, values, 3 );
	const SparseMatrixCV nablaXi2(3, numParam, indecesXi2, values, 3 );
	const SparseMatrixCV nablaXj1(3, numParam, indecesXj1, values, 3 );
	const SparseMatrixCV nablaXj2(3, numParam, indecesXj2, values, 3 );
	
	
	const Vec3d& Pi = P[sitei];
	const Vec3d& Pj = P[sitej];
	const SparseMatrixCV& nablaPi = nablaP[sitei];
	const SparseMatrixCV& nablaPj = nablaP[sitej];
	//Vec3d Pi, Pj;
	//SparseMatrixCV nablaPi, nablaPj; 
	
	// TODO be optimized 
	//Jacobian_projection( Xi1, Xi2, nablaXi1, nablaXi2, tildePi, SparseMatrixCV(3, numParam), Pi,       nablaPi );
	//Jacobian_projection( Xj1, Xj2, nablaXj1, nablaXj2, tildePj, SparseMatrixCV(3, numParam), Pj,       nablaPj );
	//cout << nablaPi.t() << endl; 
	//cout << nablaP[sitei].t() << endl; 
	//

	Vec3d Pi_prime, Pj_prime;
	SparseMatrixCV nablaPi_prime, nablaPj_prime; 
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

}

// TODO: can be parallelized 
void LevenburgMaquart::Jacobian_datacost(
	vector<double>& Jacobian_nzv, 
	vector<int>&    Jacobian_colindx, 
	vector<int>&    Jacobian_rowptr,
	vector<double>& energy_matrix )
{
	for( int site=0; site < tildaP.size(); site++ ) {
		const int& label = labelID[site]; 

		// computing datacost 
		const double datacost_i = compute_datacost_for_one( lines[label], tildaP[site] ); 

		// Computing derivative for data cost analytically
		SparseMatrixCV J_datacost = Jacobian_datacost_for_one( site ); 

		energy_matrix.push_back( sqrt(datacost_i) ); 

		int nnz;
		const double* non_zero_value = NULL;
		const int* column_index = NULL;
		const int* row_pointer = NULL; 
		J_datacost.getRowMatrixData( nnz, &non_zero_value, &column_index, &row_pointer ); 
		smart_assert( J_datacost.row()==1 && J_datacost.col()==numParam, "Number of row is not correct for Jacobian matrix" );
		for( int i=0; i<nnz; i++ ) {
			Jacobian_nzv.push_back( non_zero_value[i] );
			Jacobian_colindx.push_back( column_index[i] ); 
		}
		Jacobian_rowptr.push_back( (int) Jacobian_nzv.size() ); 
	}
}



void LevenburgMaquart::Jacobian_smoothcost_thread_func(
		vector<double>& Jacobian_nzv, 
		vector<int>&    Jacobian_colindx,  
		vector<int>&    Jacobian_rowptr, 
		vector<double>& energy_matrix,
		int site )
{
	for( int neibourIndex=0; neibourIndex<13; neibourIndex++ ) { // find it's neighbour
		// the neighbour position
		int x, y, z; 
		Neighbour26::getNeigbour( neibourIndex, 
			tildaP[site][0], 
			tildaP[site][1], 
			tildaP[site][2], 
			x, y, z ); 
		if( !labelID3d.isValid(x,y,z) ) continue; // not a valid position
		// otherwise

		int site2 = labelID3d.at(x,y,z); 
		if( site2==-1 ) continue ; // not a neighbour
		// other wise, found a neighbour

		int l1 = labelID[site];
		int l2 = labelID[site2];

		if( l1==l2 ) continue; // TODO

		double smoothcost_i_before = 0, smoothcost_j_before = 0;
		compute_smoothcost_for_pair( lines[l1], lines[l2], 
			tildaP[site], tildaP[site2], 
			smoothcost_i_before, smoothcost_j_before ); 

		// add more rows to energy_matrix according to smooth cost 
		energy_matrix.push_back( sqrt( smoothcost_i_before ) ); 
		energy_matrix.push_back( sqrt( smoothcost_j_before ) ); 

		////// Computing derivative of pair-wise smooth cost analytically
		SparseMatrixCV J[2];
		Jacobian_smoothcost_for_pair( site, site2, J[0], J[1] ); 

		for( int ji = 0; ji<2; ji++ ) {
			int nnz;
			const double* non_zero_value = NULL;
			const int * column_index = NULL;
			const int* row_pointer = NULL; 
			J[ji].getRowMatrixData( nnz, &non_zero_value, &column_index, &row_pointer ); 

			smart_assert( J[ji].row()==1 && J[ji].col()==numParam, 
				"Number of row is not correct for Jacobian matrix" );

			for( int n1=0; n1 < nnz; n1++ ) {
				Jacobian_nzv.push_back( non_zero_value[n1] );
				Jacobian_colindx.push_back( column_index[n1] ); 
			}
			Jacobian_rowptr.push_back( (int) Jacobian_nzv.size() ); 
		}

	} // for each pair of pi and pj
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
	const Data3D<int>& indeces = labelID3d; 

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
		for( int site = 0; site < tildaP.size(); site++ ) { // For each data point
			Jacobian_smoothcost_thread_func( Jacobian_nzv_loc, Jacobian_colindx_loc, 
				Jacobian_rowptr_loc, energy_matrix_loc, site ); 
		} // end of contruction of Jacobian Matrix

		// obtatin current thread id
		int tid = omp_get_thread_num(); 
		// copy the data size of the current thead to global vectors nzv_size and nzv
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
#pragma omp parallel /* Fork a team of threads*/ 
	{
		// local variables for different processes
		vector<double> Jacobian_nzv_loc;
		vector<int>    Jacobian_colindx_loc;
		vector<int>    Jacobian_rowptr_loc;
		vector<double> energy_matrix_loc;

#pragma omp for

		for( int site = 0; site < tildaP.size(); site++ ) { 
			// For each data point, the following computation will 
			// be splited into multiple thread 
			Jacobian_smoothcost_thread_func( 
				Jacobian_nzv_loc, 
				Jacobian_colindx_loc, 
				Jacobian_rowptr_loc, 
				energy_matrix_loc, site ); 
		}

#pragma omp critical 
		{
			Jacobian_nzv.insert( Jacobian_nzv.end(), 
				Jacobian_nzv_loc.begin(), Jacobian_nzv_loc.end());
			Jacobian_colindx.insert( Jacobian_colindx.end(), 
				Jacobian_colindx_loc.begin(), Jacobian_colindx_loc.end());
			
			int offset = Jacobian_rowptr.back();
			for( int i=0; i<Jacobian_rowptr_loc.size(); i++ ) {
				Jacobian_rowptr.push_back( Jacobian_rowptr_loc[i] + offset ); 
			}

			energy_matrix.insert( energy_matrix.end(), 
				energy_matrix_loc.begin(), 
				energy_matrix_loc.end() ); 
		}
	}
}


void LevenburgMaquart::Jacobian_smoothcost(
	vector<double>& Jacobian_nzv, 
	vector<int>&    Jacobian_colindx, 
	vector<int>&    Jacobian_rowptr,
	vector<double>& energy_matrix )
{
	for( int site = 0; site < tildaP.size(); site++ ) { 
		// For each data point
		// using only one thread for the computation 
		Jacobian_smoothcost_thread_func( 
			Jacobian_nzv, 
			Jacobian_colindx, 
			Jacobian_rowptr, 
			energy_matrix, 
			site ); 
	} 
}

void LevenburgMaquart::reestimate( void )
{
	const vector<Line3D*>& lines = modelset.models; 
	const Data3D<int>& indeces = labelID3d; 

	if( lines.size()==0 ) {
		cout << "No line models available" << endl;
		return; 
	}
	int numParamPerLine = lines[0]->getNumOfParameters(); 

	double energy_before = compute_energy( tildaP, labelID, lines, indeces );

	
	P = vector<Vec3d>( tildaP.size() );
	nablaP = vector<SparseMatrixCV>( tildaP.size() );
	
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
		//Jacobian_smoothcost_openmp_critical_section( Jacobian_nzv, Jacobian_colindx, Jacobian_rowptr, energy_matrix );
		Jacobian_smoothcost_openmp( Jacobian_nzv, Jacobian_colindx, Jacobian_rowptr, energy_matrix );
		//Jacobian_smoothcost( Jacobian_nzv, Jacobian_colindx, Jacobian_rowptr, energy_matrix );
		
		// Construct Jacobian matrix
		SparseMatrixCV Jacobian = SparseMatrix(
			(int) Jacobian_rowptr.size() - 1, 
			(int) lines.size() * numParamPerLine, 
			Jacobian_nzv, Jacobian_colindx, Jacobian_rowptr );
		
		// SparseMatrixCV A = Jacobian.t() * Jacobian + SparseMatrixCV::I( Jacobian.col() ) * lambda;
		SparseMatrixCV Jt_J = Jacobian.t() * Jacobian; 
		SparseMatrixCV A = Jt_J + Jt_J.diag() * lambda;

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

		double new_energy = compute_energy( tildaP, labelID, lines, indeces );

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
		vector<double> minT( (int) labelID.size(), (std::numeric_limits<double>::max)() );
		vector<double> maxT( (int) labelID.size(), (std::numeric_limits<double>::min)() );
		for( int site = 0; site < tildaP.size(); site++ ) { // For each data point
			int label = labelID[site]; 
			Vec3d p1, p2;
			lines[label]->getEndPoints( p1, p2 ); 

			Vec3d& pos = p1;
			Vec3d dir = p2 - p1; 
			dir /= sqrt( dir.dot( dir ) ); // normalize the direction 
			double t = ( Vec3d(tildaP[site]) - pos ).dot( dir );
			maxT[label] = max( t+1, maxT[label] );
			minT[label] = min( t-1, minT[label] );
		}
		for( int label=0; label < labelID.size(); label++ ) {
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

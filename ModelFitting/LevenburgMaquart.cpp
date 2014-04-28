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


void LevenburgMaquart::datacost_jacobian(
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
		SparseMatrixCV J_datacost = compute_datacost_derivative( lines[label], dataPoints[site] ); 

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
			compute_smoothcost_derivative(  lines[l1], lines[l2], 
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
				compute_smoothcost_derivative(  lines[l1], lines[l2], 
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

		int tid = omp_get_thread_num(); // obtatin current thread id
		nzv_size[tid] = (unsigned int) Jacobian_nzv_loc.size(); 
		nzv_rows[tid] = (unsigned int) Jacobian_rowptr.size(); 
#pragma omp barrier // wait for all thread to execute till this point 
		unsigned int old_nzv_size = (unsigned int) Jacobian_nzv.size();
		unsigned int old_num_rows = (unsigned int) Jacobian_rowptr.size(); 

		if( tid == omp_get_num_threads()-1 ) {
			accumulate_nzv_size[0] = nzv_size[0]; 
			accumulate_nzv_rows[0] = nzv_rows[0]; 
			for( int i=1; i<=tid; i++ ) {
				accumulate_nzv_size[i] = nzv_size[i] + accumulate_nzv_size[i-1]; 
				accumulate_nzv_rows[i] = nzv_rows[i] + accumulate_nzv_rows[i-1]; 
			}
			// resize the result vector, make them bigger!
			int nThreads = omp_get_num_threads();
			Jacobian_nzv.resize( Jacobian_nzv.size() + accumulate_nzv_size[nThreads] ); 
			Jacobian_colindx.resize( Jacobian_colindx.size() + accumulate_nzv_size[nThreads] ); 
			Jacobian_rowptr.resize( Jacobian_rowptr.size() + accumulate_nzv_rows[nThreads] ); 
			energy_matrix.resize( energy_matrix.size() + accumulate_nzv_rows[nThreads] );
		}
		
#pragma omp barrier // wait for all thread to execute till this point 
		// copy data to result vector 
		int offset = old_nzv_size + (tid>0) ? accumulate_nzv_size[tid-1] : 0; 
		memcpy( &Jacobian_nzv[0] + offset, 
			&Jacobian_nzv_loc[0], 
			accumulate_nzv_size[tid] * sizeof(double) ); 
		memcpy( &Jacobian_colindx[0] + offset, 
			&Jacobian_colindx_loc[0], 
			accumulate_nzv_size[tid] * sizeof(double) ); 

		int offset_row = old_num_rows + (tid>0) ? accumulate_nzv_rows[tid-1] : 0; 
		for( int i=0; i<Jacobian_rowptr_loc.size(); i++ ) {
			Jacobian_rowptr[i+offset_row] = Jacobian_rowptr_loc[i] + offset; 
		}

		memcpy( &energy_matrix[tid*2 + old_num_rows],
			&energy_matrix_loc[0], 
			accumulate_nzv_rows[tid] * sizeof(double) ); 
/*
#pragma omp critical 
		{
			Jacobian_nzv.insert( Jacobian_nzv.end(), Jacobian_nzv_loc.begin(), Jacobian_nzv_loc.end());
			Jacobian_colindx.insert( Jacobian_colindx.end(), Jacobian_colindx_loc.begin(), Jacobian_colindx_loc.end());
			int offset = Jacobian_rowptr.back();
			for( int i=0; i<Jacobian_rowptr_loc.size(); i++ ) {
				Jacobian_rowptr.push_back( Jacobian_rowptr_loc[i] + offset ); 
			}
			energy_matrix.push_back( energy_matrix_loc ); 
		}
*/

	}
}

void LevenburgMaquart::reestimate( void )
{
	const vector<Line3D*>& lines = modelset.models; 
	const Data3D<int>& indeces = labelIDs; 

	int numParamPerLine = lines[0]->getNumOfParameters(); 

	double energy_before = compute_energy( dataPoints, labelings, lines, indeces );

	double lambda = 1e2; // lamda - damping function for levenburg maquart
	int lmiter = 0; // levenburg maquarit iteration count
	for( lmiter = 0; lmiter<1; lmiter++ ) { 

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
		// datacost_jacobian( Jacobian_nzv, Jacobian_colindx, Jacobian_rowptr, energy_matrix );

		// // // // // // // // // // // // // // // // // // 
		// Construct Jacobian Matrix - smooth cost 
		// // // // // // // // // // // // // // // // // // 
		Timer::begin( "J smoothcost n m t" ); 
		Jacobian_smoothcost( Jacobian_nzv, Jacobian_colindx, Jacobian_rowptr, energy_matrix );
		Timer::end( "J smoothcost n m t" ); 
		//Timer::begin( "J smoothcost m t" ); 
		//Jacobian_smoothcost_openmp( Jacobian_nzv, Jacobian_colindx, Jacobian_rowptr, energy_matrix );
		//Timer::end( "J smoothcost m t" ); 

		// Construct Jacobian matrix
		SparseMatrixCV Jacobian = SparseMatrix(
			(int) Jacobian_rowptr.size() - 1, 
			(int) lines.size() * numParamPerLine, 
			Jacobian_nzv, Jacobian_colindx, Jacobian_rowptr );
		
		Jacobian.t() * Jacobian;
		SparseMatrixCV::I( Jacobian.col() ) * lambda; 
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


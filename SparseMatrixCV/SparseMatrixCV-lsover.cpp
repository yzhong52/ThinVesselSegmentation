#include "SparseMatrixCV.h"
#include <iostream>
#include <opencv2/core/core.hpp>

using namespace std;
#include "lsolver\bicgsq.h"

void mult( const SparseMatrixCV &A, const double *v, double *w ) {
	for( int i=0; i<A.row(); i++ ) ;

	int N;
	const double* non_zero_value = NULL;
	const int * column_index = NULL;
	const int* row_pointer = NULL;
	A.getRowMatrixData( N, &non_zero_value, &column_index, &row_pointer );

	// int previous = -1;
	for( int r=0; r<A.row(); r++ ) {
		w[r] = 0.0;
		for( int i = row_pointer[r]; i < row_pointer[r+1]; i++ ) {
			const int& c = column_index[i];
			w[r] += non_zero_value[i] * v[c];
		}
	}
}

void solve( const SparseMatrixCV& A, const cv::Mat_<double>& B, cv::Mat_<double>& X, SparseMatrixCV::Options o ){

	X = cv::Mat_<double>::zeros( A.row(), 1 );
	switch( o ) {
		case SparseMatrixCV::BICGSQ:
			bicgsq( A.row(), A, (double*)B.data, (double*)X.data, 1e-3 );
			break;
		case SparseMatrixCV::SUPERLU:
			X = cv::Mat_<double>::zeros( A.row(), 1 );
			solve( A, (const double*)B.data, (double*) X.data );
			break;
	}
}

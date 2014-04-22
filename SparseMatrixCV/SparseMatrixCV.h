#pragma once

#include "opencv2/core/core.hpp"
#include "../SparseMatrix/SparseMatrix.h"

// A wraper for SparseMatrix for OpenCV
class SparseMatrixCV : public SparseMatrix 
{
	
public:
	SparseMatrixCV( void ) { }
	SparseMatrixCV( int nrow, int ncol, const int index[][2], const double value[], int N );
	SparseMatrixCV( int nrow, int ncol ) { }
	SparseMatrixCV( const SparseMatrixCV& m ){ }
	SparseMatrixCV( const SparseMatrix& m ) : SparseMatrix( m ) { }
	~SparseMatrixCV( void ) { }

	SparseMatrixCV( const cv::Mat& m ) { }

	template <class _Tp, int m, int n>
	SparseMatrixCV( const cv::Matx<_Tp, m, n>& vec );

	template <class _Tp, int m, int n>
	friend SparseMatrixCV operator*( const cv::Matx<_Tp,m,n>& vec, const SparseMatrixCV& sm );
	
	friend const cv::Mat operator*( const SparseMatrixCV& m1, const cv::Mat m2 ){
		// TODO:

		return cv::Mat(); 
	}

	inline const SparseMatrixCV t() const {
		return SparseMatrix::t(); 
	}
};



template <class _Tp, int m, int n>
SparseMatrixCV::SparseMatrixCV( const cv::Matx<_Tp, m, n>& vec ) {
	vector<double> non_zero_value;
	vector<int> col_index;
	vector<int> row_pointer;

	row_pointer.push_back( 0 ); 
	for( int r = 0; r < m; r++ ) {
		for( int c = 0; c < n; c++ ) {
			if( abs( vec(r, c) )>1e-12 ) {
				non_zero_value.push_back( vec(r, c) ); 
				col_index.push_back( c ); 
			}
		}
		row_pointer.push_back( (int) non_zero_value.size() ); 
	}

	// re-constuct the matrix with give data
	this->updateData( m, n, non_zero_value, col_index, row_pointer ); 
}

template <class _Tp, int m, int n>
SparseMatrixCV operator*( const cv::Matx<_Tp,m,n>& vec, const SparseMatrixCV& sm ){
	SparseMatrixCV res; 
	return res; 
}
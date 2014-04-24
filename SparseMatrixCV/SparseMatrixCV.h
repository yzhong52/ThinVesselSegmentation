#pragma once

#include "opencv2/core/core.hpp"
#include "../SparseMatrix/SparseMatrix.h"

// A wraper for SparseMatrix for OpenCV
class SparseMatrixCV : public SparseMatrix 
{
public:
	/////////////////////////////////////////////////////////////////
	// Constructors & Destructors 
	// // // // // // // // // // // // // // // // // // // // // // 
	SparseMatrixCV( void ) : SparseMatrix( 0, 0 ){ }
	
	SparseMatrixCV( int nrow, int ncol ) : SparseMatrix( nrow, ncol ){ }
	
	SparseMatrixCV( int num_rows, int num_cols, const double non_zero_value[], const int col_index[], const int row_pointer[], int N ) 
		: SparseMatrix( num_rows, num_cols, non_zero_value, col_index, row_pointer, N ) { }
	
	SparseMatrixCV( const SparseMatrixCV& m ) : SparseMatrix( m ) { }
	
	SparseMatrixCV( const SparseMatrix& m ) : SparseMatrix( m ) { }

	SparseMatrixCV( int nrow, int ncol, const int index[][2], const double value[], int N );

	template <class _Tp>
	SparseMatrixCV( const cv::Mat_<_Tp>& m );

	~SparseMatrixCV( void ) { }

	
	/////////////////////////////////////////////////////////////////
	// Matrix manipulation functions
	// // // // // // // // // // // // // // // // // // // // // // 

	template <class _Tp, int m, int n>
	SparseMatrixCV( const cv::Matx<_Tp, m, n>& vec );

	template <class _Tp, int m, int n>
	friend SparseMatrixCV operator*( const cv::Matx<_Tp,m,n>& vec, const SparseMatrixCV& sm );
	
	friend const cv::Mat_<double> operator*( const SparseMatrixCV& m1, const cv::Mat_<double>& m2 );
	
	inline const SparseMatrixCV t() const { return SparseMatrix::t(); }

	void getRowMatrixData( int& N, double const** non_zero_value, int const** column_index, 
		int const** row_pointer ) const;

	static SparseMatrixCV I( int rows );

	void convertTo( cv::Mat_<double>& m ); 

	/////////////////////////////////////////////////////////////////
	// friends function for liner sover
	// // // // // // // // // // // // // // // // // // // // // // 
	enum Options{ BICGSQ, SUPERLU };
	friend void mult( const SparseMatrixCV &A, const double *v, double *w );
	friend void solve( const SparseMatrixCV& A, const cv::Mat_<double>& B, cv::Mat_<double>& X, 
		Options o = BICGSQ );
};



template <class _Tp, int m, int n>
SparseMatrixCV::SparseMatrixCV( const cv::Matx<_Tp, m, n>& vec ) : SparseMatrix(0,0) {
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

template <class _Tp>
SparseMatrixCV::SparseMatrixCV( const cv::Mat_<_Tp>& m ) : SparseMatrix( m.rows, m.cols ) {
	vector<double> non_zero_value;
	vector<int> col_index;
	vector<int> row_pointer;

	row_pointer.push_back( 0 ); 
	for( int r = 0; r < m.rows; r++ ) {
		for( int c = 0; c < m.cols; c++ ) {
			if( abs( m(r, c) )>1e-12 ) {
				non_zero_value.push_back( m(r, c) ); 
				col_index.push_back( c ); 
			}
		}
		row_pointer.push_back( (int) non_zero_value.size() ); 
	}

	// re-constuct the matrix with give data
	this->updateData( m.rows, m.cols, non_zero_value, col_index, row_pointer ); 
}

template <class _Tp, int m, int n>
SparseMatrixCV operator*( const cv::Matx<_Tp,m,n>& vec, const SparseMatrixCV& sm ){
	// TODO: this count be furture optimized by replacing the SparseMatrixCV( vec ) 
	// with actual computing the maltiplication within this function
	return SparseMatrixCV( vec ) * sm; 
}


#pragma once

#include "opencv2/core/core.hpp"
#include "../SparseMatrix/SparseMatrix.h"

// A wraper for SparseMatrix for OpenCV
class SparseMatrixCV : public SparseMatrix 
{
	
public:
	SparseMatrixCV( void ) { }
	SparseMatrixCV( int nrow, int ncol, const int indecesM1[][2], const double values[], int N ) { }
	SparseMatrixCV( int nrow, int ncol ) { }
	SparseMatrixCV( const SparseMatrixCV& m ){ }
	SparseMatrixCV( const SparseMatrix& m ) : SparseMatrix( m ) { }
	~SparseMatrixCV( void ) { }

	SparseMatrixCV( const cv::Mat& m ) { }

	template <class _Tp, int m, int n>
	SparseMatrixCV( const cv::Matx<_Tp, m, n>& vec ) { }

	template <class _Tp, int m, int n>
	friend SparseMatrixCV operator*( const cv::Matx<_Tp,m,n>& vec, const SparseMatrixCV& sm ){
		SparseMatrixCV res; 
		return res; 
	}
	
	friend const cv::Mat operator*( const SparseMatrixCV& m1, const cv::Mat m2 ){
		// TODO:

		return cv::Mat(); 
	}

	const SparseMatrixCV t() const{
		return SparseMatrix::t(); 
	}
};


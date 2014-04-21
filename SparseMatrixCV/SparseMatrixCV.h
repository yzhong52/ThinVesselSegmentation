#pragma once

#include "opencv2/core/core.hpp"
#include "../SparseMatrix/SparseMatrix.h"

// A wraper for SparseMatrix for OpenCV
class SparseMatrixCV
{
	SparseMatrix data; 
public:
	SparseMatrixCV( void ) : data(0,0,0,0,0,0) { }
	SparseMatrixCV( int nrow, int ncol, const int indecesM1[][2], const double values[], int N ) { }
	SparseMatrixCV( int nrow, int ncol ) : data(0,0,0,0,0,0) { }
	SparseMatrixCV( const SparseMatrixCV& m ) : data(0,0,0,0,0,0) { }
	SparseMatrixCV( const SparseMatrix& m ) : data( m ) { }
	~SparseMatrixCV( void ) { }

	SparseMatrixCV( const cv::Mat& m ) : data(0,0,0,0,0,0) { }

	template <class _Tp, int m, int n>
	SparseMatrixCV( const cv::Matx<_Tp, m, n>& vec ) : data(0,0,0,0,0,0) { }

	template <class _Tp, int m, int n>
	friend SparseMatrixCV operator*( const cv::Matx<_Tp,m,n>& vec, const SparseMatrixCV& sm ){
		SparseMatrixCV res; 
		return res; 
	}
	
	friend const cv::Mat operator*( const SparseMatrixCV& m1, const cv::Mat m2 ){
		// TODO:

		return cv::Mat(); 
	}

	friend const SparseMatrixCV operator*( const SparseMatrixCV& m1, const double& value ){
		return m1.data * value; 
	}

	friend const SparseMatrixCV operator/( const SparseMatrixCV& m1, const double& value ){
		return m1.data / value; 
	}

	friend const SparseMatrixCV operator+( const SparseMatrixCV& m1, const SparseMatrixCV& m2 ){
		return m1.data + m2.data; 
	}
	friend const SparseMatrixCV operator*( const SparseMatrixCV& m1, const SparseMatrixCV& m2 ){
		return m1.data * m2.data; 
	}
	friend const SparseMatrixCV operator/( const SparseMatrixCV& m1, const SparseMatrixCV& m2 ){
		return m1.data / m2.data; 
	}

	friend const SparseMatrixCV operator-( const SparseMatrixCV& m1, const SparseMatrixCV& m2 ){
		return m1.data - m2.data; 
	}

	const SparseMatrixCV t() const{
		return data.t(); 
	}
};


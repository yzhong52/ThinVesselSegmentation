#pragma once

#include "opencv2/core/core.hpp"
#include "../SuperMatrix/SparseMatrix.h"


class SparseMatrixCV : SparseMatrix
{
public:
	SparseMatrixCV( void )  : SparseMatrix(0,0,0,0,0,0) { }
	SparseMatrixCV( int nrow, int ncol, const int indecesM1[][2], const double values[], int N )  : SparseMatrix(0,0,0,0,0,0) { }
	SparseMatrixCV( int nrow, int ncol ) : SparseMatrix(0,0,0,0,0,0) { }
	SparseMatrixCV( const SparseMatrixCV& m ) : SparseMatrix(0,0,0,0,0,0) { }
	~SparseMatrixCV(void) { }

	SparseMatrixCV( const cv::Mat& m ) : SparseMatrix(0,0,0,0,0,0) { }

	template <class _Tp, int m, int n>
	SparseMatrixCV( const cv::Matx<_Tp, m, n>& vec ) : SparseMatrix(0,0,0,0,0,0) { }

	template <class _Tp, int m, int n>
	friend SparseMatrixCV operator*( const cv::Matx<_Tp,m,n>& vec, const SparseMatrixCV& sm ){
		SparseMatrixCV res; 
		return res; 
	}
	

	friend const SparseMatrixCV operator*( const SparseMatrixCV& m1, const double& value ){
		SparseMatrixCV res; 
		return res; 
	}

	friend const SparseMatrixCV operator/( const SparseMatrixCV& m1, const double& value ){
		SparseMatrixCV res; 
		return res; 
	}

	friend const SparseMatrixCV operator+( const SparseMatrixCV& m1, const SparseMatrixCV& m2 ){
		SparseMatrixCV res; 
		return res; 
	}
	friend const SparseMatrixCV operator*( const SparseMatrixCV& m1, const SparseMatrixCV& m2 ){
		SparseMatrixCV res; 
		return res; 
	}

	friend const cv::Mat operator*( const SparseMatrixCV& m1, const cv::Mat m2 ){
		return cv::Mat(); 
	}

	friend const SparseMatrixCV operator/( const SparseMatrixCV& m1, const SparseMatrixCV& m2 ){
		SparseMatrixCV res; 
		return res; 
	}

	friend const SparseMatrixCV operator-( const SparseMatrixCV& m1, const SparseMatrixCV& m2 )
	{
		SparseMatrixCV res; 
		return res; 
	}

	const SparseMatrixCV t() const{
		SparseMatrixCV res; 
		return res; 
	}
};


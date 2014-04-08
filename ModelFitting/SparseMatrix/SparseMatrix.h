#pragma once

// The following class dependent on three libraries
// 1. OpenCV (http://opencv.org/)
// 2. lin_solver(http://aam.mathematik.uni-freiburg.de/IAM/Research/projectskr/lin_solver/)
// 3. cblas (http://www.netlib.org/clapack/cblas/)


#include "opencv2\core\core.hpp"
#include <unordered_set>
#include <iostream>
#include "RC.h"

class SparseMatrix
{
	void init( const int& r, const int& c ); 
public:
	// constructor & destructor
	SparseMatrix( int rows = 1, int cols = 1 );
	SparseMatrix( int rows, int cols, const int indeces[][2], const double value[], int N );
	SparseMatrix( const SparseMatrix& m );
	template <class T, int n>
	SparseMatrix( const cv::Vec<T, n>& vec );
	~SparseMatrix( void );

	const SparseMatrix& operator=( const SparseMatrix& matrix ); 

	// size of the matrix
	inline int rows() const { return sm.size(0); }
	inline int cols() const { return sm.size(1); }

	// setter
	void set( const int& r, const int& c, const double& value );
	// getter 
	double get( const int& r, const int& c ) const; 

	SparseMatrix clone(void) const;

	// multiply the current matrix with a matrix
	SparseMatrix multiply( const SparseMatrix& matrix ) const;
	// multiply the current matrix with a transposed matrix
	SparseMatrix multiply_transpose( const SparseMatrix& matrix ) const;
	// transposed the current matrix and then mutiply another matrix
	SparseMatrix transpose_multiply( const SparseMatrix& matrix ) const;
	SparseMatrix transpose_multiply( const cv::Mat& matrix ) const;

	const SparseMatrix operator*( const double& value ) const; 
	const SparseMatrix operator/( const double& value ) const; 
	const SparseMatrix& operator*=( const double& value ); 
	const SparseMatrix& operator/=( const double& value ); 

	void setWithOffSet( const SparseMatrix& matrix, int offsetR, int offsetC ); 

	static SparseMatrix ones( int rows, int cols ); 

	inline cv::SparseMat getMatrix( void) { return sm; } 
private:

	cv::SparseMat sm; 

	// use reference counting
	RC* reference;

	// use a map to keep track of the non-zero indexces for each row
	std::vector< std::unordered_set<int> >* unzeros_for_row; 
	std::vector< std::unordered_set<int> >* unzeros_for_col; 
	
	// getter
	inline double at( const int& r, const int& c ) const {
		const int idx[] = {r, c}; 
		return sm.value<double>( idx ); 
	}
	// setter
	inline double& at( const int& r, const int& c ) {
		const int idx[] = {r, c}; 
		return sm.ref<double>( idx ); 
	}

	friend void mult( const SparseMatrix &A, const double *v, double *w );

	friend const SparseMatrix operator-( const SparseMatrix& m1, const SparseMatrix& m2 ); 
	friend const SparseMatrix operator+( const SparseMatrix& m1, const SparseMatrix& m2 ); 
	friend const SparseMatrix operator*( const SparseMatrix& m1, const SparseMatrix& m2 ); 
	friend const SparseMatrix operator*( const SparseMatrix& m1, const cv::Mat& m2 ); 
	friend std::ostream& operator<<( std::ostream& out, const SparseMatrix& matrix ) ;
};


template <class T, int n> 
SparseMatrix::SparseMatrix( const cv::Vec<T, n>& vec ) {
	init( n ,1 ); 
	for( int i=0; i<n; i++ ) this->set( i, 0, vec[i] ); 
}

namespace cv{
	// Overload the opencv solve function so that it can take SparseMatrix as input
	void solve( const SparseMatrix& A, const Mat& B, Mat& X );
};


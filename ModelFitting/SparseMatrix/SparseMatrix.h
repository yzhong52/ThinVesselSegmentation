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
	SparseMatrix clone(void) const;

	// size of the matrix
	inline int rows() const { return sm.size(0); }
	inline int cols() const { return sm.size(1); }

	// setter & getter
	inline void set( const int& r, const int& c, const double& value );
	inline double get( const int& r, const int& c ) const; 
	inline cv::SparseMat getMatrix( void) { return sm; } 

	// multiply the current matrix with a matrix
	SparseMatrix multiply( const SparseMatrix& matrix ) const;

	// multiply the current matrix with the transposed of anomatrix
	SparseMatrix multiply_transpose( const SparseMatrix& matrix ) const;

	// transposed the current matrix and then mutiply another matrix
	SparseMatrix transpose_multiply( const SparseMatrix& matrix ) const;
	cv::Mat      transpose_multiply( const cv::Mat& matrix ) const;

	template <class T, int n>
	friend SparseMatrix transpose_multiply( const  cv::Vec<T, n>& vec, const SparseMatrix& sm );
	template <class T, int n>
	friend SparseMatrix multiply( const  cv::Vec<T, n>& vec, const SparseMatrix& sm );

	const SparseMatrix operator*( const double& value ) const; 
	const SparseMatrix operator/( const double& value ) const; 
	const SparseMatrix& operator*=( const double& value ); 
	const SparseMatrix& operator/=( const double& value ); 

	// copy the value of a matrix m1 to the current matrix
	void setWithOffSet( const SparseMatrix& m1, int offsetR, int offsetC ); 

	// create a matrix with all ones	
	static SparseMatrix ones( int rows, int cols ); 

private:

	cv::SparseMat sm; 

	// use reference counting
	RC* reference;

	// use a map to keep track of the non-zero indexces for each row
	std::vector< std::unordered_set<int> >* unzeros_for_row; 
	std::vector< std::unordered_set<int> >* unzeros_for_col; 
	
	// getter
	inline double  at( const int& r, const int& c ) const {
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




namespace cv{
	// Overload the opencv solve function so that it can take SparseMatrix as input
	void solve( const SparseMatrix& A, const Mat& B, Mat& X );
};





////////////////////////////////////////////////////////////////////////////////////////
// SparseMatrix template functions

template <class T, int n> 
SparseMatrix::SparseMatrix( const cv::Vec<T, n>& vec ) {
	init( n ,1 ); 
	for( int i=0; i<n; i++ ) this->set( i, 0, vec[i] ); 
}

template <class T, int n>
SparseMatrix transpose_multiply( const  cv::Vec<T, n>& vec, const SparseMatrix& sm ){
	smart_assert( n==sm.rows(), "Matrix sizes do not mathc. " );

	// allocate memory
	SparseMatrix res( 1, sm.cols() ); 

	// iterator of column index in sm
	std::unordered_set<int>::iterator it; 

	// for each col in sm
	for( int c = 0; c < sm.cols(); c++ ) {
		double value = 0; 
		for( it=sm.unzeros_for_col->at(c).begin(); it != sm.unzeros_for_col->at(c).end(); it++ ) {
			value += vec[*it] * sm.get(*it, c);
		}
		res.set( 0, c, value); 
	}

	// return matrix
	return res; 
}

template <class T, int n>
SparseMatrix multiply( const  cv::Vec<T, n>& vec, const SparseMatrix& sm ){
	smart_assert( 1==sm.rows(), "Matrix sizes do not mathc. " );

	// allocate memory
	SparseMatrix res( n, sm.cols() ); 

	// iterator of column index in sm
	std::unordered_set<int>::iterator it; 

	// for each col in sm, sm has only one row
	for( it=sm.unzeros_for_row->at(0).begin(); it!=sm.unzeros_for_row->at(0).end(); it++ ){
		for( int i=0; i<n; i++ ) {
			res.set( i, *it, vec[i] * sm.get(0, *it) ); 
		}
	}

	// return matrix
	return res; 
}


////////////////////////////////////////////////////////////////////////////////////////
// Inline functions for SparseMatrix

inline void SparseMatrix::set( const int& r, const int& c, const double& value ){
	static const double epsilon = 1e-50; 
	// maintain a map of the non-zero indeces
	if( abs(value) < epsilon ) {
		unzeros_for_row->at(r).erase( c ); 
		unzeros_for_col->at(c).erase( r ); 
		// erase this value from sparse matrix
		sm.erase( r, c ); 
	} else{
		unzeros_for_row->at(r).insert( c ); 
		unzeros_for_col->at(c).insert( r ); 
		at( r, c ) = value;  // update the value
	}
}

inline double SparseMatrix::get( const int& r, const int& c) const {
	return at( r, c ); 
}

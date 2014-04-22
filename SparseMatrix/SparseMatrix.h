#pragma once

// This class is build with SuperLU 4.3, referece can be found in the following lin: 
// http://crd-legacy.lbl.gov/~xiaoye/SuperLU/

#include <iostream>
#include "SparseMatrixData.h"
#include "RC.h"
#include <vector>

// This SparseMatrix implementation, once the matrix is created
// it is almost impossible to modify the values anymore. 
// but it is designe for solving linear equations. 

class SparseMatrix
{
protected:
	// matrix data are all store in class SparseMatrixData
	SparseMatrixData *data; 
	// for reference counting 
	RC *rc; 
public:
	// initialized a zero matrix
	SparseMatrix( int num_rows = 0, int num_cols = 0 ); 
	SparseMatrix( int num_rows, int num_cols, 
		const double non_zero_value[], 
		const int col_index[], 
		const int row_pointer[], 
		int N );
		// use (const T*) to force the constructor to make a deep copy of the data
	SparseMatrix( int num_rows, int num_cols, 
		const std::vector<double> non_zero_value, 
		const std::vector<int> col_index, 
		const std::vector<int> row_pointer );
	// copy constructor 
	SparseMatrix( const SparseMatrix& matrix );
	const SparseMatrix& operator=( const SparseMatrix& matrix ); 	
	// deep copy of the matrix data
	const SparseMatrix clone(void) const;
	// destructor
	~SparseMatrix(void);

	// get the number of rows and column of the matrix
	inline const int row() const { return data->row(); } 
	inline const int col() const { return data->col(); } 

	bool updateData( int num_rows, int num_cols, 
		const std::vector<double> non_zero_value, 
		const std::vector<int> col_index, 
		const std::vector<int> row_pointer );
	bool updateData(  int num_rows, int num_cols, 
		const double non_zero_value[], 
		const int col_index[], 
		const int row_pointer[], 
		int N );
	bool updateData(  int num_rows, int num_cols, 
		double non_zero_value[], 
		int col_index[], 
		int row_pointer[], 
		int N );
	
	////////////////////////////////////////////////////////////////
	// Matrix manipulations 
	////////////////////////////////////////////////////////////////

	// Transpose a matrix
	const SparseMatrix t() const;
	// mutiply by value
	const SparseMatrix& operator*=( const double& value ); 
	const SparseMatrix& operator/=( const double& value ); 
	friend void solve( const SparseMatrix& A, const double* B, double* X ); 
	//friend const SparseMatrix multiply( const SparseMatrix& m1, const SparseMatrix& m2 ); 
	friend const SparseMatrix operator/( const SparseMatrix& m1, const SparseMatrix& m2 ); 
	friend const SparseMatrix operator*( const SparseMatrix& m1, const SparseMatrix& m2 ); 
	friend const SparseMatrix operator+( const SparseMatrix& m1, const SparseMatrix& m2 ); 
	friend const SparseMatrix operator-( const SparseMatrix& m1, const SparseMatrix& m2 ); 
	friend const SparseMatrix operator/( const SparseMatrix& m1, const double& value ); 
	friend const SparseMatrix operator*( const SparseMatrix& m1, const double& value ); 

	// utility functions
	friend std::ostream& operator<<( std::ostream& out, const SparseMatrix& m );
};

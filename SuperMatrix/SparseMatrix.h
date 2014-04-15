#pragma once

// This class is build with SuperLU 4.3, referece can be found in the following lin: 
// http://crd-legacy.lbl.gov/~xiaoye/SuperLU/

#include "SuperLU_4.3\SRC\slu_ddefs.h"

class SparseMatrix
{
	SuperMatrix* dataCol; // matrix data stored in collumn order
	SuperMatrix* dataRow; // matrix data stored in row order

	int rows; // number of rows
	int cols; // number of cols
public:
	SparseMatrix( int rows, int cols );
	SparseMatrix( int rows, int cols, const int indeces[][2], const double value[], int N );
	SparseMatrix clone(void) const;
	const SparseMatrix& operator=( const SparseMatrix& matrix ); 	
	SparseMatrix(void);
	~SparseMatrix(void);

	inline const int& row() const { return rows; } 
	inline const int& col() const { return cols; } 

	const SparseMatrix& operator*=( const double& value ); 
	const SparseMatrix& operator/=( const double& value ); 

	friend void solve( const SparseMatrix& A, const double* B, double* X ); 
	friend const SparseMatrix multiple( const SparseMatrix& m1, const SparseMatrix& m2 ); 
	friend const SparseMatrix transpose_multiple( const SparseMatrix& m1, const SparseMatrix& m2 ); 
	friend const SparseMatrix multiple_transpose( const SparseMatrix& m1, const SparseMatrix& m2 ); 
	friend const SparseMatrix add( const SparseMatrix& m1, const SparseMatrix& m2 ); 
	friend const SparseMatrix subtract( const SparseMatrix& m1, const SparseMatrix& m2 ); 
};

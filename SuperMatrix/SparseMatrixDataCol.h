#pragma once
#include "SuperLU_4.3\SRC\slu_ddefs.h"

// wrapper class for matrix data store in col order
class SparseMatrixDataCol {
	friend class SparseMatrixData; 

	// data
	SuperMatrix* supermatrix;

	// getters
	NCformat* getData() { return (NCformat*) supermatrix->Store; }
	inline double* nzvel(){  return (double*) getData()->nzval; }
	inline int*    rowinx(){ return getData()->rowind; }
	inline int*    colptr(){ return getData()->colptr; }

	// constructors & destructors
	SparseMatrixDataCol( int rows, int cols, double nzval[], int rowidx[], int colptr[], int N ); 
	SparseMatrixDataCol( int rows, int cols, const double non_zero_value[], 
		const int row_index[], const int col_pointer[], int N ); 
	~SparseMatrixDataCol(); 
}; 

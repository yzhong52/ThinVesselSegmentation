#define _CRT_SECURE_NO_WARNINGS
#include "SparseMatrixDataCol.h"

SparseMatrixDataCol::SparseMatrixDataCol( int rows, int cols, 
	const double non_zero_value[], const int row_index[], const int col_pointer[], int N )
{
	// non-zero values
	double   *nzval = NULL; 
	if ( !(nzval = doubleMalloc(N)) ) ABORT("Fail to alloc memory for SparseMatrix");
	memcpy( nzval, non_zero_value, sizeof(double) * N ); 

	int* rowind = NULL; 
	if ( !(rowind = intMalloc(N)) ) ABORT("Fail to alloc memory for SparseMatrix");
	memcpy( rowind, row_index, sizeof(int) * N ); 

	int* colptr; 
	if ( !(colptr = intMalloc(rows+1)) ) ABORT("Fail to alloc memory for SparseMatrix");
	memcpy( colptr, col_pointer, sizeof(int) * cols ); 
	colptr[cols] = N;

	supermatrix = new SuperMatrix();
	/* Create matrix A in the format expected by SuperLU. */
	dCreate_CompCol_Matrix( 
		supermatrix, 
		rows,   // number of rows
		cols,   // number of cols
		N,      // number of non zero entrires
		nzval,  // the values column wise
		rowind, // row indices of the nonzeros
		colptr, // beginning of columns in nzval[] and rowind[]
		SLU_NR, // Data storage order: row-wise
		SLU_D,  // Data type: double
		SLU_GE);// Matrix type: genral
}

SparseMatrixDataCol::SparseMatrixDataCol( int rows, int cols, double nzval[], 
	int rowind[], int colptr[], int N )
{
	supermatrix = new SuperMatrix();
	/* Create matrix A in the format expected by SuperLU. */
	dCreate_CompCol_Matrix( 
		supermatrix, 
		rows,   // number of rows
		cols,   // number of cols
		N,      // number of non zero entrires
		nzval,  // the values column wise
		rowind, // row indices of the nonzeros
		colptr, // beginning of columns in nzval[] and rowind[]
		SLU_NR, // Data storage order: row-wise
		SLU_D,  // Data type: double
		SLU_GE);// Matrix type: genral
}


SparseMatrixDataCol::~SparseMatrixDataCol(){
	if( supermatrix ) {
		Destroy_CompCol_Matrix( supermatrix );
		supermatrix = NULL; 
	}
}


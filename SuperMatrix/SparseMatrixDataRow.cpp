#define _CRT_SECURE_NO_WARNINGS
#include "SparseMatrixDataRow.h"

SparseMatrixDataRow::SparseMatrixDataRow( int rows, int cols, 
	const double non_zero_value[], const int col_index[], const int row_pointer[], int N )
{
	// non-zero values
	double   *nzval = NULL; 
	if ( !(nzval = doubleMalloc(N)) ) ABORT("Fail to alloc memory for SparseMatrix");
	memcpy( nzval, non_zero_value, sizeof(double) * N ); 

	int* colind = NULL; 
	if ( !(colind = intMalloc(N)) ) ABORT("Fail to alloc memory for SparseMatrix");
	memcpy( colind, col_index, sizeof(int) * N ); 

	int* rowptr; 
	if ( !(rowptr = intMalloc(rows+1)) ) ABORT("Fail to alloc memory for SparseMatrix");
	memcpy( rowptr, row_pointer, sizeof(int) * rows ); 
	rowptr[rows] = N;

	supermatrix = new SuperMatrix();
	/* Create matrix A in the format expected by SuperLU. */
	dCreate_CompCol_Matrix( 
		supermatrix, 
		rows,   // number of rows
		cols,   // number of cols
		N,      // number of non zero entrires
		nzval,  // the values column wise
		colind, // row indices of the nonzeros
		rowptr, // beginning of columns in nzval[] and rowind[]
		SLU_NC, // Data storage order: column-wise
		SLU_D,  // Data type: double
		SLU_GE);// Matrix type: genral
}

SparseMatrixDataRow::SparseMatrixDataRow( int rows, int cols, double nzval[], 
	int colidx[], int rowptr[], int N )
{
	supermatrix = new SuperMatrix();
	/* Create matrix A in the format expected by SuperLU. */
	dCreate_CompCol_Matrix( 
		supermatrix, 
		rows,   // number of rows
		cols,   // number of cols
		N,      // number of non zero entrires
		nzval,  // the values column wise
		colidx, // row indices of the nonzeros
		rowptr, // beginning of columns in nzval[] and rowind[]
		SLU_NC, // Data storage order: column-wise
		SLU_D,  // Data type: double
		SLU_GE);// Matrix type: genral
} 


SparseMatrixDataRow::~SparseMatrixDataRow(){
	if( supermatrix ) {
		Destroy_CompRow_Matrix( supermatrix );
		supermatrix = NULL; 
	}
}

#pragma once
#include "SparseMatrixDataCol.h"
#include "SparseMatrixDataRow.h"

//#include "SuperLU_4.3\SRC\slu_ddefs.h"

//// wrapper class for matrix data store in row order
//class SparseMatrixDataRow {
//	friend class SparseMatrixData; 
//
//	// data
//	SuperMatrix* supermatrix;
//
//	// getters
//	NRformat* getData() { return (NRformat*) supermatrix->Store; }
//	inline double* nzvel(){  return (double*) getData()->nzval; }
//	inline int*    colinx() { return getData()->colind; }
//	inline int*    rowptr() { return getData()->rowptr; }
//
//	// constructors & destructors
//	SparseMatrixDataRow( int rows, int cols, double nzval[], int colidx[], int rowptr[], int N ); 
//	SparseMatrixDataRow( int rows, int cols, const double non_zero_value[], 
//		const int col_index[], const int row_pointer[], int N ); 
//	~SparseMatrixDataRow(); 
//}; 

//// wrapper class for matrix data store in col order
//class SparseMatrixDataCol {
//	friend class SparseMatrixData; 
//
//	// data
//	SuperMatrix* supermatrix;
//
//	// getters
//	NCformat* getData() { return (NCformat*) supermatrix->Store; }
//	inline double* nzvel(){  return (double*) getData()->nzval; }
//	inline int*    rowinx() { return getData()->rowind; }
//	inline int*    colptr() { return getData()->colptr; }
//
//	// constructors & destructors
//	SparseMatrixDataCol( int rows, int cols, double nzval[], int rowidx[], int colptr[], int N ); 
//	SparseMatrixDataCol( int rows, int cols, const double non_zero_value[], 
//		const int row_index[], const int col_pointer[], int N ); 
//	~SparseMatrixDataCol(); 
//}; 

class SparseMatrixData
{
	// data
	SparseMatrixDataCol* datacol; // matrix data stored in collumn order
	SparseMatrixDataRow* datarow; // matrix data stored in row order

public:
	// constructor
	~SparseMatrixData(){
		if( datacol ) delete datacol;
		if( datarow ) delete datarow; 
	}

	inline SuperMatrix* getCol(){ 
		if( datarow==NULL || datacol ) return datacol->supermatrix; 

		/* Convert the compressed row fromat to the compressed column format. */
		double* nzval = NULL; 
		int *rowidx = NULL;
		int *colptr = NULL; 
		dCompRow_to_CompCol( 
			datarow->supermatrix->nrow, // number of rows
			datarow->supermatrix->ncol, // number of cols
			datarow->getData()->nnz,    // number of non-zero entries
			datarow->nzvel(),           // non-zero entries
			datarow->colinx(),          // column index
			datarow->rowptr(),          // row pointers
			&nzval,                     // non-zero entries (for column-order matrix)
			&rowidx,                    // row indeces
			&colptr );                  // column pointers

		datacol = new SparseMatrixDataCol( 
			datarow->supermatrix->nrow,
			datarow->supermatrix->ncol,
			nzval,
			rowidx,
			colptr,
			datarow->getData()->nnz );
		return datacol->supermatrix;
	}

	inline SuperMatrix* getRow(){
		if( datacol==NULL || datarow ) return datarow->supermatrix;

		/* Convert the compressed row fromat to the compressed column format. */
		double* nzval = NULL; 
		int *colidx = NULL;
		int *rowptr = NULL; 
		dCompRow_to_CompCol( 
			datacol->supermatrix->nrow, // number of rows
			datacol->supermatrix->ncol, // number of cols
			datacol->getData()->nnz,    // number of non-zero entries
			datacol->nzvel(),           // non-zero entries
			datacol->rowinx(),          // column index
			datacol->colptr(),          // row pointers
			&nzval,                     // non-zero entries (for row-order matrix)
			&colidx,                    // column indeces
			&rowptr );                  // column pointers

		datarow = new SparseMatrixDataRow( 
			datarow->supermatrix->nrow,
			datarow->supermatrix->ncol,
			nzval,
			colidx,
			rowptr,
			datacol->getData()->nnz );
		return datarow->supermatrix;
	}
};


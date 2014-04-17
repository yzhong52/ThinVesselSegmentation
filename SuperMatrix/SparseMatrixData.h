#pragma once
#include "SparseMatrixDataCol.h"
#include "SparseMatrixDataRow.h"
#include <utility>
#include <iostream> 
class SparseMatrixData
{
	SparseMatrixDataCol* datacol; // matrix data stored in collumn order
	SparseMatrixDataRow* datarow; // matrix data stored in row order

	int ncol, nrow; 
public:
	// constructor & destructor 
	// By defaut the is stored as row order
	SparseMatrixData( int num_rows, int num_cols, const double non_zero_value[], 
		const int col_index[], const int row_pointer[], int N ) : ncol( num_cols ), nrow( num_rows )
	{
		datarow = new SparseMatrixDataRow( num_rows, num_cols, non_zero_value, col_index, row_pointer, N ); 
		datacol = NULL; 
	}

	~SparseMatrixData(){
		if( datacol ) delete datacol;
		if( datarow ) delete datarow; 
	}

	const int col() const { return ncol; }
	const int row() const { return nrow; }

	const SparseMatrixDataCol* const getCol(){ 
		if( datarow==NULL || datacol ) return datacol; 

		/* Convert the compressed row fromat to the compressed column format. */
		double* nzval = NULL; 
		int *rowidx = NULL;
		int *colptr = NULL; 
		dCompRow_to_CompCol( 
			nrow, // number of rows
			ncol, // number of cols
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
		return datacol;
	}

	const SparseMatrixDataRow* const getRow(){
		if( datacol==NULL || datarow ) return datarow;

		/* Convert the compressed row fromat to the compressed column format. */
		double* nzval = NULL; 
		int *colidx = NULL;
		int *rowptr = NULL; 

		// IMPORTANT: ncol and nrow is reversed in the following function. 
		// This is because dCompRow_to_CompCol is a function for converting 
		// row-order sparse matrix to col-order matrix. 
		dCompRow_to_CompCol( 
			ncol,						// number of rows
			nrow,						// number of cols
			datacol->getData()->nnz,    // number of non-zero entries
			datacol->nzvel(),           // non-zero entries
			datacol->rowinx(),          // column index
			datacol->colptr(),          // row pointers
			&nzval,                     // non-zero entries (for row-order matrix)
			&colidx,                    // column indeces
			&rowptr );                  // row pointers

		datarow = new SparseMatrixDataRow( 
			nrow,
			ncol,
			nzval,
			colidx,
			rowptr,
			datacol->getData()->nnz );
		return datarow;
	}

	void transpose( void ) {
		// tanspose number of rows and columns
		std::swap( nrow, ncol ); 
		
		if( datarow && datacol ) {
			std::swap( datarow->supermatrix->Store, datacol->supermatrix->Store ); 
			if( datarow ) std::swap( datarow->supermatrix->ncol, datarow->supermatrix->nrow ); 
			if( datacol ) std::swap( datacol->supermatrix->ncol, datacol->supermatrix->nrow ); 
		} 
		else if( datarow ) 
		{
			datacol = new SparseMatrixDataCol( 
				nrow,   // number of rows
				ncol,   // number of cols
				datarow->nzvel(),    // non-zero values
				datarow->colinx(),   // row index
				datarow->rowptr(),   // col pointer
				datarow->getData()->nnz );
			
			delete datarow->supermatrix; 
			datarow->supermatrix = NULL; 
			delete datarow; 
			datarow = NULL; 
		} 
		else if( datacol ) 
		{
			datarow = new SparseMatrixDataRow( 
				nrow,   // number of rows
				ncol,   // number of cols
				datacol->nzvel(),    // non-zero values
				datacol->rowinx(),   // column index
				datacol->colptr(),   // row pointer
				datacol->getData()->nnz );

			delete datacol->supermatrix; 
			datacol->supermatrix = NULL; 
			delete datacol; 
			datacol = NULL; 
		} 
		else 
		{
			// this is a empty matrix, nothing to transpose
		}
	}
};


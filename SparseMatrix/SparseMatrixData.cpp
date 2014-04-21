#define _CRT_SECURE_NO_WARNINGS
#include "SparseMatrixData.h"


SparseMatrixData::SparseMatrixData( int num_rows, int num_cols, const double non_zero_value[], 
	const int col_index[], const int row_pointer[], int N ) 
	: ncol( num_cols ), nrow( num_rows ), datarow( NULL ), datacol( NULL )
{
	if( N != 0 ) {
		datarow = new SparseMatrixDataRow( num_rows, num_cols, non_zero_value, col_index, row_pointer, N ); 
	}
	// eles if N==0, then this is a zero matrix, then don't allocate matrix data
}


SparseMatrixData::~SparseMatrixData(){
	// destro matrix data
	if( datacol ) delete datacol;
	if( datarow ) delete datarow; 
}

const SparseMatrixDataCol* const SparseMatrixData::getCol(){ 
	if( datarow==NULL || datacol ) return datacol; 

	// memory of the following arrays will be allocated in func dCompRow_to_CompCol
	double* nzval = NULL; 
	int *rowidx = NULL;
	int *colptr = NULL; 

	dCompRow_to_CompCol( 
		nrow,						// number of rows
		ncol,						// number of cols
		datarow->getData()->nnz,    // number of non-zero entries
		datarow->nzvel(),           // non-zero entries
		datarow->colinx(),          // column index
		datarow->rowptr(),          // row pointers
		&nzval,                     // non-zero entries (for column-order matrix)
		&rowidx,                    // row indeces
		&colptr );                  // column pointers

	datacol = new SparseMatrixDataCol( nrow, ncol, nzval, rowidx, colptr, datarow->getData()->nnz );

	return datacol;
}

const SparseMatrixDataRow* const SparseMatrixData::getRow(){
	if( datacol==NULL || datarow ) return datarow;

	// memory of the following arrays will be allocated in func dCompRow_to_CompCol
	double* nzval = NULL; 
	int *colidx = NULL;
	int *rowptr = NULL; 

	// IMPORTANT!: ncol and nrow is reversed in the following function. 
	// This is because dCompRow_to_CompCol is a function for converting 
	// row-order sparse matrix to col-order matrix. But it can also be 
	// used to comvert a col matrix into a row one if carefully used. 
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

	datarow = new SparseMatrixDataRow( nrow, ncol, nzval, colidx, rowptr, datacol->getData()->nnz );

	return datarow;
}

void SparseMatrixData::transpose( void ) {
	// tanspose number of rows and columns
	std::swap( nrow, ncol ); 

	if( datarow && datacol ) {
		// if the matrix is stored as both row-order and col-order
		std::swap( datarow->supermatrix->Store, datacol->supermatrix->Store ); 
		if( datarow ) std::swap( datarow->supermatrix->ncol, datarow->supermatrix->nrow ); 
		if( datacol ) std::swap( datacol->supermatrix->ncol, datacol->supermatrix->nrow ); 
	} 
	else if( datarow ) 
	{
		// if the matrix is stored as row-order only
		// convert the matrix from row-order to col-order
		datacol = new SparseMatrixDataCol( 
			nrow,   // number of rows
			ncol,   // number of cols
			datarow->nzvel(),    // non-zero values
			datarow->colinx(),   // row index
			datarow->rowptr(),   // col pointer
			datarow->getData()->nnz );

		// destroy row-order data 
		// IMPORTANT: the matrix data stored in supermatrix won't be destroyed. 
		//  as they are used in datacol->sparsematrix
		delete datarow->supermatrix;    datarow->supermatrix = NULL; 
		delete datarow;                 datarow = NULL; 
	} 
	else if( datacol ) 
	{
		// if the matrix is stored as col-order only
		// convert the matrix from col-order to row-order
		datarow = new SparseMatrixDataRow( 
			nrow,   // number of rows
			ncol,   // number of cols
			datacol->nzvel(),    // non-zero values
			datacol->rowinx(),   // column index
			datacol->colptr(),   // row pointer
			datacol->getData()->nnz );

		// destroy col-order data
		// IMPORTANT: the matrix data stored in supermatrix won't be destroyed. 
		//  as they are used in datarow->sparsematrix
		delete datacol->supermatrix;    datacol->supermatrix = NULL; 
		delete datacol;                 datacol = NULL; 
	} 
	else 
	{
		// both datacol and datarow are NULL
		// this is a empty matrix, nothing to transpose
	}
}

void SparseMatrixData::multiply( const double& value ){
	if( datarow ) {
		for( int i=0; i<datarow->nnz(); i++ ) datarow->nzvel()[i] *= value; 
	}
	if( datacol ) {
		for( int i=0; i<datacol->nnz(); i++ ) datacol->nzvel()[i] *= value; 
	}
}
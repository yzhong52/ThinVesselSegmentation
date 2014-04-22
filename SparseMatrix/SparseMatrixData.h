#pragma once
#include "SparseMatrixDataCol.h"
#include "SparseMatrixDataRow.h"
#include <utility>



class SparseMatrixData
{
	// size of the matrix
	int ncol, nrow; 

	SparseMatrixDataCol* datacol; // matrix data stored in collumn order
	SparseMatrixDataRow* datarow; // matrix data stored in row order
public:
	// constructor & destructor 
	// By defaut the is stored as row order
	SparseMatrixData( 
		int num_rows,                   // number of row
		int num_cols,                   // number of cols
		const double non_zero_value[],  // non-zero values
		const int col_index[],			// pointer to column indeces
		const int row_pointer[],		// pointers to data of each row
		int N );						// number of non-zero values
	SparseMatrixData( 
		int num_rows,                   // number of row
		int num_cols,                   // number of cols
		double non_zero_value[],		// non-zero values
		int col_index[],				// pointer to column indeces
		int row_pointer[],				// pointers to data of each row
		int N );						// number of non-zero values
	// create an empty matrix
	SparseMatrixData( int num_rows, int num_cols);
	~SparseMatrixData();

	inline const int col() const { return ncol; }
	inline const int row() const { return nrow; }
	inline const bool isZero ( void ) const { return !datacol && !datarow; }

	const SparseMatrixDataCol* const getCol();
	const SparseMatrixDataRow* const getRow();
	void transpose( void );
	void multiply( const double& value );
};


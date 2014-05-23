#pragma once
#include "SparseMatrixDataCol.h"
#include "SparseMatrixDataRow.h"
#include <utility>


#include <string.h>


/*! \brief Convert a row compressed storage into a column compressed storage.
 */
inline void dCompRow_to_CompCol_yuchen(int m, int n, int nnz,
		    double *a, int *colind, int *rowptr,
		    double **at, int **rowind, int **colptr)
{
    register int i, j, col, relpos;
    int *marker;

    /* Allocate storage for another copy of the matrix. */
    *at = new double[nnz];
    *rowind = new int[nnz];
    *colptr = new int[n+1]; // (int *) intMalloc(n+1);
    marker = new int[n]; // (int *) intCalloc(n);
    memset(marker, 0, sizeof(int)*n );

    /* Get counts of each column of A, and set up column pointers */
    for (i = 0; i < m; ++i)
	for (j = rowptr[i]; j < rowptr[i+1]; ++j) ++marker[colind[j]];
    (*colptr)[0] = 0;
    for (j = 0; j < n; ++j) {
        (*colptr)[j+1] = (*colptr)[j] + marker[j];
        marker[j] = (*colptr)[j];
    }

    /* Transfer the matrix into the compressed column storage. */
    for (i = 0; i < m; ++i) {
        for (j = rowptr[i]; j < rowptr[i+1]; ++j) {
            col = colind[j];
            relpos = marker[col];
            (*rowind)[relpos] = i;
            (*at)[relpos] = a[j];
            ++marker[col];
        }
    }

    delete[] marker;
}


class SparseMatrixData
{
public:
	// size of the matrix
	int ncol, nrow;

	SparseMatrixDataCol* datacol; // matrix data stored in collumn order
	SparseMatrixDataRow* datarow; // matrix data stored in row order
public:
    // create an zero matrix
	SparseMatrixData( unsigned num_rows, unsigned num_cols);

	// constructor & destructor
	// By defaut the is stored as row order
	SparseMatrixData(
		int num_rows,                   // number of row
		int num_cols,                   // number of cols
		const double non_zero_value[],  // non-zero values
		const int col_index[],			// pointer to column indeces
		const int row_pointer[],		// pointers to data of each row
		int N );						// number of non-zero values


	~SparseMatrixData();

	inline const int& col() const { return ncol; }
	inline const int& row() const { return nrow; }
	inline bool isZero ( void ) const { return !datacol && !datarow; }

	SparseMatrixDataCol* getCol();
	SparseMatrixDataRow* getRow();
	void transpose( void );
	void multiply( const double& value );
};


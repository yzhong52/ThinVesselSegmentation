#pragma once
#include <utility>
#include <string.h>

class SparseMatrixData
{
    class MatrixData
    {
        friend class SparseMatrixData;
    private:
        int nnz;       // number of non-zero value
        double *nzval; // pointer to array of nonzero values
        union
        {
            // row order representation
            struct
            {
                int *colind; // pointer to array of columns indices of the nonzeros
                int *rowptr; // pointer to array of beginning of rows in nzval[] and colind[]
            };
            // column order representtaion
            struct
            {
                int *rowind; // pointer to array of row indices of the nonzeros
                int *colptr; // pointer to array of beginning of columns in nzval[], and rowind[]
            };
        };

        MatrixData()
            : nnz(0), nzval(nullptr), colind(nullptr), rowptr(nullptr) {}

        inline void release()
        {
            delete[] nzval;
            delete[] colind;
            delete[] rowptr;
            clear();
        }

        inline void clear()
        {
            nnz = 0;
            nzval = nullptr;
            colind = nullptr;
            rowptr = nullptr;
        }

        bool isEmpty() const
        {
            return nnz==0;
        }
    };
public:
    // size of the matrix
    int ncol, nrow;

    MatrixData datacol; // matrix data stored in collumn order
    MatrixData datarow; // matrix data stored in row order
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

    // dtor
    ~SparseMatrixData();

    inline const int& col() const
    {
        return ncol;
    }
    inline const int& row() const
    {
        return nrow;
    }
    inline bool isZero ( void ) const
    {
        return isRow() && isCol();
    }
    inline bool isRow (void) const
    {
        return datarow.isEmpty();
    }
    inline bool isCol (void) const
    {
        return datacol.isEmpty();
    }
    void getCol(int& N, const double*& nzval, const int *&rowind, const int*& colptr );
    void getRow(int& N, const double*& nzval, const int *&colind, const int*& rowptr );

    void transpose( void );
    void multiply( const double& value );

private:
    void RowMatrix_to_ColMatrix(int row, int col, int nnz,
                                double *a, int *colind, int *rowptr,
                                double **at, int **rowind, int **colptr);
};


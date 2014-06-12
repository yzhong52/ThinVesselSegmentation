#include <opencv2/core/core.hpp>
#include <iostream>
#include "SparseMatrix.h"

using namespace std;

// Please refer to the following link for more details about the following library
// http://aam.mathematik.uni-freiburg.de/IAM/Research/projectskr/lin_solver/
#include "lsolver/bicgsq.h"

void mult( const SparseMatrix &A, const double *v, double *w )
{
    unsigned N = 0;
    const double* non_zero_value = NULL;
    const unsigned* column_index = NULL;
    const unsigned* row_pointer = NULL;
    A.getRowMatrixData( N, non_zero_value, column_index, row_pointer );

    for( unsigned r=0; r<A.row(); r++ )
    {
        w[r] = 0.0;
        for( unsigned i = row_pointer[r]; i < row_pointer[r+1]; i++ )
        {
            const unsigned& c = column_index[i];
            w[r] += non_zero_value[i] * v[c];
        }
    }
}

void solve( const SparseMatrix& A, const double* B, double* X,
           double acuracy, SparseMatrix::Options o )
{
    switch( o )
    {
    case SparseMatrix::BICGSQ:
        bicgsq( A.row(), A, B, X, acuracy );
        break;
    case SparseMatrix::SUPERLU:

        break;
    }
}

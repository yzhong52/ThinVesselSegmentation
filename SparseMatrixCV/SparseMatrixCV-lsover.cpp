#include <opencv/cv.h>
#include <iostream>
#include "SparseMatrixCV.h"

using namespace std;

// Please refer to the following link for more details about the following library
// http://aam.mathematik.uni-freiburg.de/IAM/Research/projectskr/lin_solver/
#include "lsolver/bicgsq.h"

void mult( const SparseMatrixCV &A, const double *v, double *w )
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

void solve( const SparseMatrixCV& A, const cv::Mat_<double>& B, cv::Mat_<double>& X,
           double acuracy, SparseMatrixCV::Options o )
{
    X = cv::Mat_<double>::zeros( A.row(), 1 );
    switch( o )
    {
    case SparseMatrixCV::BICGSQ:
        bicgsq( A.row(), A, (double*)B.data, (double*)X.data, acuracy );
        break;
    case SparseMatrixCV::SUPERLU:

        break;
    }
}

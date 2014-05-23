#include <opencv2/core/core.hpp>

#include "SparseMatrixCV.h"
#include <iostream>

using namespace std;

// Please refer to the following link for more details about the following library
// http://aam.mathematik.uni-freiburg.de/IAM/Research/projectskr/lin_solver/
#include "lsolver\bicgsq.h"

void mult( const SparseMatrixCV &A, const double *v, double *w )
{
    for( int i=0; i<A.row(); i++ ) ;

    int N;
    const double* non_zero_value = NULL;
    const int * column_index = NULL;
    const int* row_pointer = NULL;
    A.getRowMatrixData( N, &non_zero_value, &column_index, &row_pointer );

    // int previous = -1;
    for( int r=0; r<A.row(); r++ )
    {
        w[r] = 0.0;
        for( int i = row_pointer[r]; i < row_pointer[r+1]; i++ )
        {
            const int& c = column_index[i];
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

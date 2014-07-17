#include "SparseMatrixCV.h"
#include <vector>
using namespace std;

// The following not supported by g++
#if _MSC_VER && !__INTEL_COMPILER
#ifdef _DEBUG
#pragma comment(lib,"../x64/Debug/SparseMatrix.lib")
#else
#pragma comment(lib,"../x64/Release/SparseMatrix.lib")
#endif
#endif

SparseMatrixCV::SparseMatrixCV( unsigned nrow, unsigned ncol, const unsigned index[][2], const double value[], unsigned N )
    : SparseMatrix(0,0)
{
    struct Three
    {
        Three( unsigned r, unsigned c, double v) : row(r), col(c), val(v) { }
        unsigned row, col;
        double val;
        inline bool operator<( const Three& t ) const
        {
            return row < t.row || (row==t.row && col < t.col);
        }
    };
    vector<Three> initdata;
    for( unsigned i=0; i<N; i++ )
    {
        initdata.push_back( Three( index[i][0], index[i][1], value[i] ) );
    }

    std::sort( initdata.begin(), initdata.end() );


    vector<double> non_zero_value(N);
    vector<unsigned> col_index(N);
    vector<unsigned> row_pointer(nrow+1);

    int previous_row = -1;
    for( unsigned i=0; i<N; i++ )
    {
        non_zero_value[i] = initdata[i].val;
        col_index[i] = initdata[i].col;
        if( (int) initdata[i].row != previous_row )
        {
            row_pointer[++previous_row] = i;
        }
    }
    row_pointer[nrow] = N;

    // re-constuct the matrix with give data
    this->updateData( nrow, ncol, non_zero_value, col_index, row_pointer );

    // we don't need to release the data, the memroy is used in the matrix data
    //delete[] row_pointer;
    //delete[] col_index;
    //delete[] non_zero_value;
}


const cv::Mat_<double> operator*( const SparseMatrixCV& m1, const cv::Mat_<double>& m2 )
{
    cv::Mat_<double> res( m1.row(), m2.cols );

    assert( m1.col()==(unsigned)m2.rows && "Matrix size does not match" );

    std::vector<double> res_nzval;
    std::vector<unsigned> res_colidx;
    std::vector<unsigned> res_rowptr;

    unsigned N1 = 0;
    const double* nzval1 = nullptr;
    const unsigned* colidx1   = nullptr;
    const unsigned* rowptr1   = nullptr;
    m1.getRowMatrixData( N1, nzval1, colidx1, rowptr1 );

    // store the result as row-order
    res_rowptr.push_back( 0 );
    for( unsigned r=0; r < m1.row(); r++ )
    {
        for( unsigned c=0; c < (unsigned) m2.cols; c++ )
        {
            res(r,c) = 0.0;
            for( unsigned i = rowptr1[r]; i!=rowptr1[r+1]; i++ )
            {
                res(r,c) += nzval1[i] * m2( colidx1[i], c );
            }
        }
    }
    return res;
}



SparseMatrixCV SparseMatrixCV::I( unsigned rows )
{
    double* nzv = new double[rows];
    unsigned* colind = new unsigned[rows];
    unsigned* rowptr = new unsigned[rows+1];
    for( unsigned i=0; i<rows; i++ )
    {
        nzv[i] = 1.0;
        colind[i] = i;
        rowptr[i] = i;
    }
    rowptr[rows] = rows;

    return SparseMatrixCV( rows, rows, nzv, colind, rowptr, rows );
}

void SparseMatrixCV::convertTo( cv::Mat_<double>& m )
{
    m = cv::Mat_<double>::zeros( this->row(), this->col() );

    if( this->isZero() )
    {
        return;
    }

    unsigned N = 0;
    const double* nzval = nullptr;
    const unsigned* colidx   = nullptr;
    const unsigned* rowptr   = nullptr;
    this->getRowMatrixData( N, nzval, colidx, rowptr );

    unsigned vi = 0;
    for( unsigned r=0; r < this->row(); r++ )
    {
        for( unsigned c=0; c < this->col(); c++ )
        {
            if( colidx[vi]==c && vi<rowptr[r+1] )
                m(r,c) = nzval[vi++];
        }
    }
}


void solve( const SparseMatrixCV& A, const cv::Mat_<double>& B,
            cv::Mat_<double>& X, double acuracy,
            SparseMatrix::Options o )
{
    X = cv::Mat_<double>::zeros( A.row(), 1 );

    solve(A, (double*)B.data, (double*)X.data, acuracy, o);
}


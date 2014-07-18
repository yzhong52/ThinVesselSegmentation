#pragma once

#include <opencv2/core/core.hpp>
#include "../SparseMatrix/SparseMatrix.h"
#include <vector>

// A wraper for SparseMatrix for OpenCV
class SparseMatrixCV : public SparseMatrix
{
public:
    /////////////////////////////////////////////////////////////////
    // Constructors & Destructors
    // // // // // // // // // // // // // // // // // // // // // //
    SparseMatrixCV( void ) : SparseMatrix( 0, 0 ) { }

    SparseMatrixCV( unsigned nrow, unsigned ncol ) : SparseMatrix( nrow, ncol ) { }

    SparseMatrixCV( unsigned num_rows, unsigned num_cols,
                    const double non_zero_value[], const unsigned col_index[],
                    const unsigned row_pointer[], unsigned N )
        : SparseMatrix( num_rows, num_cols, non_zero_value, col_index, row_pointer, N ) { }

    SparseMatrixCV( const SparseMatrixCV& m ) : SparseMatrix( m ) { }

    SparseMatrixCV( const SparseMatrix& m ) : SparseMatrix( m ) { }

    SparseMatrixCV( unsigned nrow, unsigned ncol, const unsigned index[][2], const double value[], unsigned N );

    template <class _Tp>
    SparseMatrixCV( const cv::Mat_<_Tp>& m );

    ~SparseMatrixCV( void ) { }


    /////////////////////////////////////////////////////////////////
    // Matrix manipulation functions
    // // // // // // // // // // // // // // // // // // // // // //

    template <class _Tp, int m, int n>
    SparseMatrixCV( const cv::Matx<_Tp, m, n>& vec );

    template <class _Tp, int m, int n>
    friend SparseMatrixCV operator*( const cv::Matx<_Tp,m,n>& vec, const SparseMatrixCV& sm );

    friend const cv::Mat_<double> operator*( const SparseMatrixCV& m1, const cv::Mat_<double>& m2 );

    inline const SparseMatrixCV t() const
    {
        return SparseMatrix::t();
    }

    static SparseMatrixCV I( unsigned rows );

    void convertTo( cv::Mat_<double>& m );

    // Solving linear system
    friend void solve( const SparseMatrixCV& A,
                       const cv::Mat_<double>& B,
                       cv::Mat_<double>& X,
                       double acuracy = 1e-3,
                       SparseMatrix::Options o = BICGSQ );
};



template <class _Tp, int m, int n>
SparseMatrixCV::SparseMatrixCV( const cv::Matx<_Tp, m, n>& vec ) : SparseMatrix(0,0)
{
    std::vector<double>   non_zero_value;
    std::vector<unsigned> col_index;
    std::vector<unsigned> row_pointer;

    row_pointer.push_back( 0 );
    for( int r = 0; r < m; r++ )
    {
        for( int c = 0; c < n; c++ )
        {
            if( vec(r, c)>1e-12 || vec(r, c)<-1e-12 )
            {
                non_zero_value.push_back( vec(r, c) );
                col_index.push_back( c );
            }
        }
        row_pointer.push_back( (int) non_zero_value.size() );
    }

    // re-construct the matrix with give data
    this->updateData( m, n, non_zero_value, col_index, row_pointer );
}

template <class _Tp>
SparseMatrixCV::SparseMatrixCV( const cv::Mat_<_Tp>& m ) : SparseMatrix( m.rows, m.cols )
{
    std::vector<double > non_zero_value;
    std::vector<unsigned > col_index;
    std::vector<unsigned > row_pointer;

    row_pointer.push_back( 0 );
    for( int r = 0; r < m.rows; r++ )
    {
        for( int c = 0; c < m.cols; c++ )
        {
            if( std::abs( m(r, c) )>1e-12 )
            {
                non_zero_value.push_back( m(r, c) );
                col_index.push_back( c );
            }
        }
        row_pointer.push_back( (unsigned) non_zero_value.size() );
    }

    // re-construct the matrix with give data
    this->updateData( m.rows, m.cols, non_zero_value, col_index, row_pointer );
}

template <class _Tp, int m, int n>
SparseMatrixCV operator*( const cv::Matx<_Tp,m,n>& vec, const SparseMatrixCV& sm )
{
    // TODO: this count be future optimized by replacing the SparseMatrixCV( vec )
    // with actual computing the multiplication within this function
    return SparseMatrixCV( vec ) * sm;
}


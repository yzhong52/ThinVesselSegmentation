#include "SparseMatrix.h"

#include <iostream>
using namespace std; 

#include "lsolver/cghs.h"
#include "lsolver/bicgsq.h"
#include "lsolver/bicgstab.h"
#include "lsolver/gmres.h"

#include "smart_assert.h"

static const double epsilon = 1e-50; 

void SparseMatrix::init( const int& rows, const int& cols )
{
	int dims = 2; // Array dimensionality
	const int _sizes[] = {rows, cols}; 
	sm = cv::SparseMat( dims, _sizes, CV_64F );
	unzeros_for_row = new std::vector<std::unordered_set<int>>( rows ); 
	unzeros_for_col = new std::vector<std::unordered_set<int>>( cols ); 
	reference = new RC(); 
}

SparseMatrix::SparseMatrix(int rows, int cols)
{
	init( rows, cols ); 
}

SparseMatrix::SparseMatrix( const SparseMatrix& matrix ) 
	: sm( matrix.sm )
	, unzeros_for_row ( matrix.unzeros_for_row )
	, unzeros_for_col ( matrix.unzeros_for_col )
	, reference( matrix.reference ) 
{
	// increase reference
	reference->AddRef(); 
}

const SparseMatrix& SparseMatrix::operator=( const SparseMatrix& matrix ){
	this->~SparseMatrix(); 
	this->sm = matrix.sm;
	this->unzeros_for_row = matrix.unzeros_for_row;
	this->unzeros_for_col = matrix.unzeros_for_col;
	this->reference = matrix.reference;
	reference->AddRef(); 
	return (*this); 
}



SparseMatrix::SparseMatrix( int rows, int cols, const int indeces[][2], const double value[], int N )
{
	init( rows, cols ); 

	// initialize the data
	for( int i=0; i<N; i++ ) {
		this->set( indeces[i][0], indeces[i][1], value[i] ); 
	}
}

SparseMatrix::~SparseMatrix(void)
{
	// Decrement the reference count
	// if reference become zero delete the data
	if(reference->Release() == 0)
	{
		delete unzeros_for_row;
		delete unzeros_for_col;
		delete reference;
	}
}


void SparseMatrix::set( const int& r, const int& c, const double& value ){
	// maintain a map of the non-zero indeces
	if( abs(value) < epsilon ) {
		unzeros_for_row->at(r).erase( c ); 
		unzeros_for_col->at(c).erase( r ); 
		// erase this value from sparse matrix
		const int idx[] = {r, c}; 
		sm.erase( r, c ); 
	} else{
		unzeros_for_row->at(r).insert( c ); 
		unzeros_for_col->at(c).insert( r ); 
		at( r, c ) = value;  // update the value
	}
}

double SparseMatrix::get( const int& r, const int& c) const {
	return at( r, c ); 
}

SparseMatrix SparseMatrix::clone(void) const {
	SparseMatrix matrix( this->rows(), this->cols() ); 
	// deep copy the datas
	*(matrix.unzeros_for_row) = *(this->unzeros_for_row);
	*(matrix.unzeros_for_col) = *(this->unzeros_for_col);
	matrix.sm = this->sm.clone();
	return matrix; 
}

void mult( const SparseMatrix &A, const double *v, double *w ) {
	std::unordered_set<int>::iterator it;
	for ( int i=0; i<A.rows(); ++i ) {
		w[i] = 0; 
		for( it = A.unzeros_for_row->at(i).begin(); it != A.unzeros_for_row->at(i).end(); it++ ) {
			w[i] += A.get( i, *it ) * v[*it]; 
		} 
	}
}


const SparseMatrix operator-( const SparseMatrix& m1, const SparseMatrix& m2 ){
	smart_assert( m1.cols()==m2.cols() && m1.rows()==m2.rows(), "Matrix sizes do not mathc. " );

	SparseMatrix res = m1.clone(); 
	
	// iterator of column index in matrix 1 and matrix 2
	std::unordered_set<int>::iterator it2; 

	// for each row in matrix 1
	for ( int i=0; i<m1.rows(); ++i ) {
		// for each column in row i of matrix 2
		for( it2 = m2.unzeros_for_row->at(i).begin(); it2 != m2.unzeros_for_row->at(i).end(); it2++ ) {
			res.set( i, *it2, res.get(i,*it2) - m2.get(i,*it2) ); 
		}
	}

	return res; 
}


const SparseMatrix operator+( const SparseMatrix& m1, const SparseMatrix& m2 ){
	smart_assert( m1.cols()==m2.cols() && m1.rows()==m2.rows(), "Matrix sizes do not mathc. " );

	SparseMatrix res = m1.clone(); 
	
	// iterator of column index in matrix 1 and matrix 2
	std::unordered_set<int>::iterator it2; 

	// for each row in matrix 1
	for ( int i=0; i<m1.rows(); ++i ) {
		// for each column in row i of matrix 2
		for( it2 = m2.unzeros_for_row->at(i).begin(); it2 != m2.unzeros_for_row->at(i).end(); it2++ ) {
			res.set( i, *it2, res.get(i,*it2) + m2.get(i,*it2) ); 
		}
	}

	return res; 
} 


const SparseMatrix operator*( const SparseMatrix& m1, const SparseMatrix& m2 ){
	smart_assert( m1.cols()==m2.rows(), "Matrix sizes do not mathc. " );

	SparseMatrix res( m1.rows(), m2.cols() ); 

	// iterator of column index in matrix 1 and matrix 2
	std::unordered_set<int>::iterator it1, it2; 

	// for each row in matrix 1
	for ( int i=0; i<m1.rows(); ++i ) {
		// for each column in row i of matrix 1
		for( it1 = m1.unzeros_for_row->at(i).begin(); it1 != m1.unzeros_for_row->at(i).end(); it1++ ) {
			// the non-zero columns in matrix 1
			const int& j = *it1; 
			// for each row j in matrix 2
			for( it2 = m2.unzeros_for_row->at(j).begin(); it2 != m2.unzeros_for_row->at(j).end(); it2++ ) {
				double value = m1.get( i, j ) * m2.get( j, *it2 ) + res.get( i, j ); 
				res.set( i, j, value ); 
			}
		} 
	}
	return res; 
} 

const SparseMatrix SparseMatrix::operator/( const double& value ) const {
	SparseMatrix res = this->clone(); 
	return res /= value; 
}

const SparseMatrix SparseMatrix::operator*( const double& value ) const {
	SparseMatrix res = this->clone(); 
	return res *= value; 
}

const SparseMatrix& SparseMatrix::operator*=( const double& value )
{
	// for each row in matrix 1
	std::unordered_set<int>::iterator it1; 
	for ( int i=0; i < this->rows(); ++i ) {
		// for each column in row i of matrix 2
		for( it1 = this->unzeros_for_row->at(i).begin(); it1 != this->unzeros_for_row->at(i).end(); it1++ ) {
			this->at( i, *it1 ) *= value; 
		}
	}
	return (*this); 
} 

const SparseMatrix& SparseMatrix::operator/=( const double& value )
{
	// for each row in matrix 1
	std::unordered_set<int>::iterator it1; 
	for ( int i=0; i < this->rows(); ++i ) {
		// for each column in row i of matrix 2
		for( it1 = this->unzeros_for_row->at(i).begin(); it1 != this->unzeros_for_row->at(i).end(); it1++ ) {
			this->at( i, *it1 ) /= value; 
		}
	}
	return (*this); 
}

void SparseMatrix::setWithOffSet( const SparseMatrix& matrix, int offsetR, int offsetC ){
	smart_assert( this->cols() >= matrix.cols() + offsetC, "Destination matrix does not have enought columns. " );
	smart_assert( this->rows() >= matrix.rows() + offsetR, "Destination matrix does not have enought columns. " );

	int r, thisR; 
	std::unordered_set<int>::iterator it; 
	for( r=0, thisR = offsetR; r < matrix.rows(); r++, thisR++ ) {
		for( it = matrix.unzeros_for_row->at( r ).begin(); it != matrix.unzeros_for_row->at( r ).end(); it++ ) {
			const int& c = *it; 
			const int thisJ = c + offsetC; 
			this->set( thisR, thisJ, matrix.get(r, c) ); 
		}
	}
}

SparseMatrix SparseMatrix::multiply( const SparseMatrix& matrix ) const {
	return (*this) * matrix; 
}

SparseMatrix SparseMatrix::multiply_transpose( const SparseMatrix& matrix ) const{
	smart_assert( this->cols()==matrix.cols(), "Matrix sizes do not mathc. " );

	SparseMatrix res( this->rows(), matrix.rows() ); 

	// iterator of column index in matrix 1 and matrix 2
	std::unordered_set<int>::iterator it1, it2; 

	// for each row in the current matrix (this)
	for ( int r1 = 0; r1 < this->rows(); ++r1 ) {
		// for each column in row r of the current matrix (this)
		for( it1 = this->unzeros_for_row->at(r1).begin(); it1 != this->unzeros_for_row->at(r1).end(); it1++ ) {
			// the non-zero columns in the current matrix (this)
			const int& c1 = *it1; 

			// If matrix is transposed, each col c in matrix 2
			for( it2 = matrix.unzeros_for_col->at(c1).begin(); it2 != matrix.unzeros_for_col->at(c1).end(); it2++ ) {
				const int& r2 = *it2; 
				double value = this->get( r1, c1 ) * matrix.get( r2, c1 ) + res.get( r1, r2 ); 
				res.set( r1, r2, value ); 
			}
		} 
	}

	return res;
}

SparseMatrix SparseMatrix::transpose_multiply( const SparseMatrix& matrix ) const{
	smart_assert( this->rows()==matrix.rows(), "Matrix sizes do not mathc. " );

	SparseMatrix res( this->cols(), matrix.cols() ); 

	// iterator of column index in matrix 1 and matrix 2
	std::unordered_set<int>::iterator it1, it2; 

	// for each col in the current matrix (this)
	for ( int c1 = 0; c1 < this->cols(); ++c1 ) {
		// for each row in the current matrix (this)
		for( it1 = this->unzeros_for_col->at(c1).begin(); it1 != this->unzeros_for_col->at(c1).end(); it1++ ) {
			// the non-zero columns of the current matrix (this) in column c1
			const int& r1 = *it1; 

			// for each row j in matrix 2
			for( it2 = matrix.unzeros_for_row->at(r1).begin(); it2 != matrix.unzeros_for_row->at(r1).end(); it2++ ) {
				const int& c2 = *it2; 
				double value = this->get( r1, c1 ) * matrix.get( r1, c2 ) + res.get( c1, c2 ); 
				res.set( c1, c2, value ); 
			}
		} 
	}

	return res; 
}

SparseMatrix SparseMatrix::ones( int rows, int cols ){
	SparseMatrix matrix( rows, cols ); 
	for( int r=0; r<rows; r++ ) for( int c=0; c<cols; c++ ) {
		matrix.set( r, c, 1.0 ); 
	}
	return matrix; 
}


std::ostream& operator<<( std::ostream& out, const SparseMatrix& matrix ){
	for( int i=0; i < matrix.rows(); i++ ) {
		for( int j=0; j < matrix.cols(); j++ ) {
			cout.width( 3 ); 
			cout << matrix.get(i, j ) << " ";
		}
		cout << endl; 
	}
	return out; 
}

namespace cv{
	// Overload the opencv solve function so that it can take SparseMatrix as input
	void solve( const SparseMatrix& A, const Mat& B, Mat& X ){
		smart_assert( B.type()==CV_64F, "Data type does not match. " );
		smart_assert(B.rows== A.cols() && B.cols==1, "Data type does not match. " );

		X = Mat::zeros(1, A.cols(), CV_64F ); 
		int its = bicgstab( A.rows(), A, (double*)B.data, (double*)X.data, 1e-5 );
	}
};



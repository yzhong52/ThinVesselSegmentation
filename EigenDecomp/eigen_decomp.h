#pragma once

/* Compute the eigen value decomposition of a 3 by 3 symetric matrix
Input:
A = [ A11  A12  A13     = [ A[0]  A[1]  A[2]
A21  A22  A23          0    A[3]  A[4]
A31  A32  A33 ];       0     0    A[5] ];
Output:
eigenvalues, eigenvectors

Reference: http://en.wikipedia.org/wiki/Eigenvalue_algorithm#3.C3.973_matrices
*/
#include <cmath>
#include <assert.h>
#include <algorithm>
#include <iostream>


#if !defined(SmallEpsilon) && !defined(BigEpsilon)
	#define SmallEpsilon 1e-18
	#define BigEpsilon 1e-4
#endif

/////////////////////////////////////////////////////////////////
// Definitions
// // // // // // // // // // // // // // // // // // // // // //

// eigenvalue decomposiiton of a three by three symmetric matrix
template<class T>
void eigen_decomp( const T A[6], T eigenvalues[3], T eigenvectors[3][3] );

// give a 3 by 3 symetric matrix and its eigenvalues, compute the corresponding eigenvectors
template<class T>
inline void eigenvector_from_value( const T A[6], const T& eig1, const T& eig2, T eigenvector[3] );

// normalize a vector
template<class T>
inline void normalize( T v[3] );

// compute the cross product of two vectors
template<class T>
inline void cross_product( const T v1[3], const T v2[3], T v3[3] );

// given a vector v1, compute two arbitratry normal vectors v2 and v3
template<class T>
void normal_vectors(  const T v1[3], T v2[3], T v3[3] ) ;


/////////////////////////////////////////////////////////////////
// Implementations
// // // // // // // // // // // // // // // // // // // // // //


template<class T>
inline void normalize( T v[3] ){
	T length = 0;
	for( int i=0; i<3; i++ ) {
		length += v[i] * v[i];
	}
	assert( length > SmallEpsilon && "Cannot normalized zero vector" );

	length = sqrt( length );
	for( int i=0; i<3; i++ ) {
		v[i] /= length;
	}
}

template<class T>
inline void cross_product( const T v1[3], const T v2[3], T v3[3] )
{
	v3[0] = v1[1] * v2[2] - v1[2] * v2[1];
	v3[1] = v1[2] * v2[0] - v1[0] * v2[2];
	v3[2] = v1[0] * v2[1] - v1[1] * v2[0];
	normalize( v3 );
}

template<class T>
void normal_vectors(  const T v1[3], T v2[3], T v3[3] ) {
	// an arbitrary combination of the following two vectors
	// is a normal to v1
	// alpha * (-v1[1], v1[0], 0) + beta * (0, -v1[2], v[1])

	if( abs(v1[0])>BigEpsilon || abs(v1[1])>BigEpsilon ) {
		v2[0] = -v1[1];
		v2[1] =  v1[0];
		v2[2] =  0;
	} else {
		v2[0] =  0;
		v2[1] = -v1[2];
		v2[2] =  v1[1];
	}

	normalize( v2 );

	cross_product( v1, v2, v3 );
}


template<class T>
void eigen_decomp( const T A[6], T eigenvalues[3], T eigenvectors[3][3] )
{
	const T& A11 = A[0];
	const T& A12 = A[1];
	const T& A13 = A[2];
	const T& A22 = A[3];
	const T& A23 = A[4];
	const T& A33 = A[5];

	T& eig1 = eigenvalues[0];
	T& eig2 = eigenvalues[1];
	T& eig3 = eigenvalues[2];

	T p1 = A12 * A12 + A13 * A13 + A23 * A23;
	if( p1 < SmallEpsilon ) { // if A is diagonal
		eig1 = A11;
		eig2 = A22;
		eig3 = A33;
		memset( eigenvectors, 0, sizeof(T) * 9 );
		eigenvectors[0][0] = eigenvectors[1][1] = eigenvectors[2][2] = 1;
	} else { // if A is not diagonal
		// Compute 1/3 of the trace of matrix A: trace(A)/3
		T q = ( A11 + A22 + A33 ) / 3;
		T p2 = (A11-q)*(A11-q) + (A22-q)*(A22-q) + (A33-q)*(A33-q) + 2 * p1;
		T p = sqrt( p2 / 6);

		// Construct matrix B
		// B = (1 / p) * (A - q * I), where I is the identity matrix
		T B11 = (1 / p) * (A11-q);
		T B12 = (1 / p) * A12;		T& B21 = B12;
		T B13 = (1 / p) * A13;		T& B31 = B13;
		T B22 = (1 / p) * (A22-q);
		T B23 = (1 / p) * A23;		T& B32 = B23;
		T B33 = (1 / p) * (A33-q);

		// Determinant of a 3 by 3 matrix B
		// Reference: http://www.mathworks.com/help/aeroblks/determinantof3x3matrix.html
		T detB = B11*(B22*B33-B23*B32) - B12*(B21*B33-B23*B31) + B13*(B21*B32-B22*B31);

		// In exact arithmetic for a symmetric matrix  -1 <= r <= 1
		// but computation error can leave it slightly outside this range.
		T r = detB / 2;
		T phi;
		const T M_PI3 = T(3.14159265 / 3);
		if( r <= -1.0f ) {
			phi = M_PI3;
		} else if (r >= 1.0f)
			phi = 0;
		else {
			phi = acos(r) / 3;
		}

		// The eigenvalues satisfy eig3 <= eig2 <= eig1
		// Notice that: trace(A) = eig1 + eig2 + eig3
		eig1 = q + 2 * p * cos( phi );
		eig3 = q + 2 * p * cos( phi + 2 * M_PI3 );
		eig2 = 3 * q - eig1 - eig3;

		// make sure that |eig1| >= |eig2| >= |eig3|
		if( abs(eig1) < abs(eig2) ) std::swap(eig1, eig2);
		if( abs(eig2) < abs(eig3) ) std::swap(eig2, eig3);
		if( abs(eig1) < abs(eig2) ) std::swap(eig1, eig2);

		// If eig1 is too small, it is not worth to compute the eigenvectors
		if( abs(eig1) < BigEpsilon ) {
			memset( eigenvectors, 0, sizeof(T) * 9 );
			eigenvectors[0][0] = eigenvectors[1][1] = eigenvectors[2][2] = 1;
		}

		// Compute the corresponding eigenvectors
		// Reference: [Cayley-Hamilton_theorem](http://en.wikipedia.org/wiki/Cayley-Hamilton_theorem)
		// If eig1, eig2, eig3 are the distinct eigenvalues of the matrix A;
		// that is: eig1 != eig2 != eig3. Then
		// (A - eig1 * I)(A - eig2 * I)(A - eig3 * I) = 0
		// Thus the columns of the product of any two of these matrices
		// will contain an eigenvector for the third eigenvalue.
		else if( abs(eig1-eig2)<BigEpsilon && abs(eig2-eig3)<BigEpsilon ) {
			memset( eigenvectors, 0, sizeof(T) * 9 );
			eigenvectors[0][0] = eigenvectors[1][1] = eigenvectors[2][2] = 1;
		} else if ( abs(eig1-eig2)<BigEpsilon ) {
			// tested
			eigenvector_from_value( A, eig1, eig2, eigenvectors[2] );
			normal_vectors( eigenvectors[2], eigenvectors[1], eigenvectors[0] );
		} else if ( abs(eig2-eig3)<BigEpsilon ) {
			// tested
			eigenvector_from_value( A, eig2, eig3, eigenvectors[0] );
			normal_vectors( eigenvectors[0], eigenvectors[1], eigenvectors[2] );
		} else {
			// eig1!=eig2 && eig2!=eig3 && eig1!=eig3
			// tested
			eigenvector_from_value( A, eig2, eig3, eigenvectors[0] );
			eigenvector_from_value( A, eig1, eig3, eigenvectors[1] );
			eigenvector_from_value( A, eig1, eig2, eigenvectors[2] );
		}
	}
}


template<class T>
inline void eigenvector_from_value( const T A[6],
	const T& eig1, const T& eig2, T eigenvector[3] )
{
	const T& A11 = A[0];
	const T& A12 = A[1];
	const T& A13 = A[2];
	const T& A22 = A[3];
	const T& A23 = A[4];
	const T& A33 = A[5];

	T tempLength = 0, length = 0;
	T tempVector[3];
	T colum[3];
	for( int i=0; i<3; i++ ) {
		switch( i ) {
		case 0:
			colum[0] = A11 - eig2;
			colum[1] = A12;
			colum[2] = A13;
			break;
		case 1:
			colum[0] = A12;
			colum[1] = A22 - eig2;
			colum[2] = A23;
			break;
		default:
			colum[0] = A13;
			colum[1] = A23;
			colum[2] = A33 - eig2;
			break;
		}
		tempVector[0] = colum[0] * (A11 - eig1) + colum[1] * A12 + colum[2] * A13;
		tempVector[1] = colum[0] * A12 + colum[1] * (A22 - eig1) + colum[2] * A23;
		tempVector[2] = colum[0] * A13 + colum[1] * A23 + colum[2] * (A33 - eig1);

		tempLength = 0;
		for( int k=0; k<3; k++ ) {
			tempLength += tempVector[k] * tempVector[k];
		}

		// There are three columns in each resulting matrix of ( A-lambda_i * I ) * ( A-lambda_j * I )
		// We choose the one with the maximum length for the sake of accuracy.
		if( length < tempLength ) {
			length = tempLength;
			memcpy( eigenvector, tempVector, sizeof(T)*3);
		}

		if( length > BigEpsilon ) break;
	}

	// The vector is almost zero,
	assert( length >= SmallEpsilon && "All eigenvector are zero. " );

	// TODO: for debuging
	if( length < SmallEpsilon ) {
		std::cout << "Lenght of vector is too small. Maybe problematic. " << std::endl;
		system("pause");
	}

	// nromailized the vector
	length = sqrt( length );
	for( int k=0; k<3; k++ ) eigenvector[k] /= length;
}

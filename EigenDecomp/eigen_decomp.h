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
#include <iostream>

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
	
	eigenvector[0] = ( A11 - eig1 ) * ( A11 - eig2 )
		+ A12 * A12 + A13 * A13; 
	eigenvector[1] = A12 * ( A11 - eig2 )
		+ (A22 - eig1) * A12 + A23 * A13; 
	eigenvector[2] = A13 * ( A11 - eig2 )
		+ A23 * A12 + (A33 - eig1) * A13; 

	T length = 0; 
	for( int i=0; i<3; i++ ) {
		length += eigenvector[i] * eigenvector[i]; 
	}
	length = sqrt( length ); 
	for( int i=0; i<3; i++ ) {
		eigenvector[i] /= length; 
	}
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
	if( p1 < 1e-6 ) { // if A is diagonal
		eig1 = A11;
		eig2 = A22;
		eig3 = A33;
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
		const T M_PI3 = 3.14159265 / 3;
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

		// Compute the corresponding eigenvectors
		// Reference: [Cayley-Hamilton_theorem](http://en.wikipedia.org/wiki/Cayley-Hamilton_theorem)
		// If eig1, eig2, eig3 are the distinct eigenvalues of the matrix A; 
		// that is: eig1 != eig2 != eig3. Then 
		// (A - eig1 * I)(A - eig2 * I)(A - eig3 * I) = 0
		// Thus the columns of the product of any two of these matrices 
		// will contain an eigenvector for the third eigenvalue. 
		if( eig1!=eig2 && eig2!=eig3 && eig1!=eig3 ) {
			eigenvector_from_value( A, eig2, eig3, eigenvectors[0] ); 
			eigenvector_from_value( A, eig1, eig3, eigenvectors[1] ); 
			eigenvector_from_value( A, eig1, eig2, eigenvectors[2] ); 
		}
	}
}
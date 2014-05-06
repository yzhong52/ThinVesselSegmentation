#include "eigen_decomp.h"
#include <iostream>
#include <cmath>

using namespace std;

int main( void ) {
	cout << "Hello World" << endl;
	/*
		A = [ 1, 2, 3; 
		      2, 4, 5;
			  3, 5, 3; ];
	*/
	double A[6] = { 1, 2, 3, 4, 5, 3 }; 
	double eigenvalues[3] = { 0 }; 
	double eigenvectors[3][3] = { 0 }; 
	eigen_decomp( A, eigenvalues, eigenvectors );
	for( int i=0; i<3; i++ ) {
		cout.width(8);
		cout << eigenvalues[i] << ": ";
		for( int j=0; j<3; j++ ){
			cout.width(10);
			cout << eigenvectors[i][j]<< " ";
		}
		cout << endl; 
	}
	
	

	// Validating A * V = lamda * V
	bool isCorrect = true; 
	cout << endl << "Validating..."; 
	double A_dense[3][3] = {
		A[0], A[1], A[2], 
		A[1], A[3], A[4],
		A[2], A[4], A[5]
	}; 
	for( int i=0; i<3; i++ ){
		for( int j=0; j<3; j++ ) {
			double v1 = 0; 
			for( int k=0; k<3; k++ ) {
				v1 += A_dense[j][k] * eigenvectors[i][k]; 
			}
			double v2 = eigenvalues[i] * eigenvectors[i][j]; 
			if( abs(v1 - v2) > 1e-4 ) {
				cout << "Result is incorrect. " << endl; 
				isCorrect = false; 
			}
		}
	}
	cout << "Result is " << (isCorrect ? "valide." : "invalid")  << endl << endl;

	return 0; 
}
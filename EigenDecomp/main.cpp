#include <iostream>
#include <cmath>
#include <vector>

#include "eigen_decomp.h"

#define EPS 1e-5

using namespace std;

int main( void ) {
	const int NUM = 8;
	double testcases[NUM][6] = {
		/* [ 1, 2, 3; 
	         2, 4, 5;
	         3, 5, 3; ];*/
		{ 1, 2, 3, 4, 5, 3 }, 
		/* [ 1, 1, 1; 
	         1, 1, 1;
	         1, 1, 1; ];*/
		{ 1, 1, 1, 1, 1, 1 },
		{ 1, 0, 0, 1, 0, 1 },
		{ 1, 0, 0, 2, 0, 3 },

		// Eigenvalues of the following matrix is 1, 1, 2
		{1.00498649389345e+000, -69.6821389976381e-003, -10.2970032930957e-003, 1.97375041442659e+000, 143.892127426565e-003, 1.02126309167995e+000}, 
		// Eigenvalues of the following matrix is -4, 1, 1
		{-1.38275915880493e+000, 533.760620310474e-003, -2.43953974867426e+000, 880.432565439340e-003, 546.480010248921e-003, -1.49767340663441e+000}, 
		// Eigenvalues of the following matrix is 3, 7, 3
		{3.74176807855796e+000, 1.55101005899563e+000, -105.925581228646e-003, 6.24310559141657e+000, -221.486535670261e-003, 3.01512633002547e+000}, 
		// Eigenvalues of the following matrix is 3, 7, 3
		{4.08520889833985e+000, 1.02353317694110e+000,1.81233236412442e+000,-2.85059962034233e+000 ,374.888222923070e-003, 5.76539072200248e+000}
	}; 

	for( int i=0; i<NUM; i++ ) {
		// if( i!=NUM-1 ) continue; 

		double* A = testcases[i]; 
		double eigenvalues[3] = { 0 }; 
		double eigenvectors[3][3] = { 0 }; 
		eigen_decomp( A, eigenvalues, eigenvectors );

		// Output result 
		cout << "Result --=+> " << endl;
		cout.width(14);
		cout << "eigenvalues " << ": eigenvectors " << endl ;
		for( int i=0; i<3; i++ ) {
			cout.width(14);
			cout << std::left << eigenvalues[i] << ": ";
			for( int j=0; j<3; j++ ){
				cout.width(14);
				cout << std::left << eigenvectors[i][j]<< " ";
			}
			cout << endl; 
		}
		
		// validate result with: A * V = lamda * V
		bool isCorrect = true; 
		cout << "Validating..." << endl; 
		double A_dense[3][3] = {
			A[0], A[1], A[2], 
			A[1], A[3], A[4],
			A[2], A[4], A[5]
		}; 
		for( int i=0; i<3; i++ ){ // for each eigenvalues
			for( int j=0; j<3; j++ ) {
				double v1 = 0; 
				for( int k=0; k<3; k++ ) {
					v1 += A_dense[j][k] * eigenvectors[i][k]; 
				}
				double v2 = eigenvalues[i] * eigenvectors[i][j]; 
				if( abs(v1 - v2) > EPS ) {
					cout << "A * V != lamda * V. " << endl; 
					isCorrect = false; 
				}
			}
		}
		// validate result with: V(i) * V(j) = 1
		for( int i=0; i<3; i++ ) {
			for( int j=i+1; j<3; j++ ) {
				double dot = 0; 
				for( int k=0; k<3; k++ ) {
					dot += eigenvectors[i][k] * eigenvectors[j][k]; 
				}
				if( abs(dot) > EPS ) {
					cout << "Vectors are not perpendicular to each other. " << endl; 
					isCorrect = false; 
				}
			}
		}
		cout << "Result is " << (isCorrect ? "valide :)" : "invalid!!!!!!!!!!!!!!!!!")  << endl << endl;
	}
	return 0; 
}
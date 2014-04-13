#include <iostream>
#include "SparseMatrix.h"
#include "opencv2\core\core.hpp"

using namespace std;
using namespace cv; 

int main( void ) { 

	// Solving linear system with build in OpenCV functions
	Mat DenseA = ( Mat_<double>(3,3) << 
		1, 0, 4, 
		0, 4, 7, 
		1, 2, 0 );
	Mat DenseB = ( Mat_<double>(3,1) << 4, 2, 1 ); 
	Mat DenseX( 3, 1, CV_64F ); 
	cv::solve( DenseA, DenseB, DenseX, DECOMP_LU ); 
	
	cout << "Sovling linear system using dense matrix = " << endl;
	cout << DenseX << endl << endl; 

	// Solving linear system with SparseMatrix
	SparseMatrix SparseA( 3, 3 );	
	for( int i=0; i<DenseA.rows; i++ ) for( int j=0; j<DenseA.cols; j++ ) {
		SparseA.set(i, j, DenseA.at<double>(i, j) );
	}
	
	solve( SparseA, DenseB, DenseX ); 
	
	cout << "Sovling linear system using sparse matrix = " << endl;
	cout << DenseX << endl << endl; 

	// Validating Reference Counting 
	SparseMatrix SparseA2 = SparseA; 
	// TODO: How to write test case for reference counting? 
	// ...
	
	// validating sums and 
	int indecesM1[][2] = { {0, 0}, {1, 2}, {2,1}, {1, 3} }; 
	double valuesM1[] = { 1.0, 3.0, 4.0, 5.0 }; 
	SparseMatrix SparseM1( 3, 4, indecesM1, valuesM1, 4 );

	cout << "SparseM1 = " << endl; 
	cout << SparseM1 << endl; 

	static const int indecesM2[][2] = { {0, 0}, {1, 2}, {2, 3} }; 
	const double valuesM2[] = { 1.0, 2.0, -2.0 }; 
	SparseMatrix SparseM2( 3, 4, indecesM2, valuesM2, 3 );
	cout << "SparseM2 = " << endl; 
	cout << SparseM2 << endl; 

	SparseMatrix SparseM3 = SparseM1 + SparseM2; 
	cout << "SparseM3 = SparseM1 + SparseM2 = " << endl; 
	cout << SparseM3 << endl;

	SparseMatrix SparseM4 = SparseM1 - SparseM2; 
	cout << "SparseM4 = SparseM1 - SparseM2 = " << endl; 
	cout << SparseM4 << endl; 

	SparseMatrix SparseM5 = SparseM1.transpose_multiply( SparseM2 ); 
	cout << "SparseM6 = SparseM1.t() * SparseM2 = " << endl; 
	cout << SparseM5 << endl; 

	Mat DenseM2;
	SparseM2.getMatrix().convertTo( DenseM2, CV_64F );
	Mat SparseM55 = SparseM1.transpose_multiply( DenseM2 );
	cout << "SparseM55 = SparseM1.t() * DenseM2 = " << endl; 
	cout << SparseM55 << endl; 


	SparseMatrix SparseM6 = SparseM1.multiply_transpose( SparseM2 ); 
	cout << "SparseM6 = SparseM1 * SparseM2.t() = " << endl; 
	cout << SparseM6 << endl; 

	SparseMatrix SparseM7 = SparseM1 * 4.0 ; 
	cout << "SparseM7 = SparseM1 * 4 = " << endl; 
	cout << SparseM7 << endl; 

	SparseMatrix SparseM8 = SparseM1 / 2.0 ; 
	cout << "SparseM8 = SparseM1 / 4 = " << endl; 
	cout << SparseM8 << endl; 

	SparseMatrix SparseM9 = SparseM1.clone(); 
	SparseM9 /= 2.0; 
	cout << "SparseM9 = SparseM1.clone(); SparseM9 /= 2.0 " << endl; 
	cout << SparseM9 << endl; 

	SparseMatrix SparseM10 = SparseM1.clone(); 
	SparseM10 *= 2.0; 
	cout << "SparseM10 = SparseM1.clone(); SparseM9 /= 2.0 " << endl; 
	cout << SparseM10 << endl; 

	SparseMatrix SparseM11( SparseM1.rows() * 2, SparseM1.cols() ); 
	SparseM11.setWithOffSet( SparseM1, 0, 0 ); 
	SparseM11 *= 2.0; 
	SparseM11.setWithOffSet( SparseM1, SparseM1.rows(), 0 ); 
	cout << "SparseM11 = SparseM1 * 0.5; SparseM11.push_back( SparseM1 ); " << endl; 
	cout << SparseM11 << endl; 

	return 0; 
}

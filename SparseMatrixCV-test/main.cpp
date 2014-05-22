//#ifdef _DEBUG
//#pragma comment(lib,"../x64/Debug/SparseMatrixCV.lib")
//#pragma comment(lib,"../x64/Debug/SparseMatrix.lib")
//#else
//#pragma comment(lib,"../x64/Release/SparseMatrixCV.lib")
//#pragma comment(lib,"../x64/Release/SparseMatrix.lib")
//#endif


#include "../SparseMatrixCV/SparseMatrixCV.h"
#include <opencv2\core\core.hpp>
#include <iostream>
using namespace cv;
using namespace std;



int main( void ) {
    cv::Mat m(3, 2, CV_32F);
    m.at<float>(0,0) = 3;
    cout << m << endl;
    cout << "hello world" << endl;

    m = cv::Mat_<double>::zeros( 4, 4 );

    cout << m << endl;

    cout << "hello world" << endl;

	cout << "############################ Identity Matrix ###########################" << endl << endl;
	cout << SparseMatrixCV::I( 4 ) << endl;


	cout << "############################ Add & Sub #################################" << endl << endl;
	{
		static const int index[][2] = { {0, 0}, {1, 1}, {2, 2} };
		static const double value[] = { 1.0, 1.0, 1.0 };
		static const SparseMatrixCV A( 3, 6, index, value, 3 );
		cout << A << endl;

		static const int indexAA[][2] = { {0, 3}, {1, 4}, {2, 5} };
		static const double valueAA[] = { 1.0, 1.0, 1.0 };
		static const SparseMatrixCV AA( 3, 6, indexAA, valueAA, 3 );
		cout << AA << endl;

		cout << A - AA << endl;
		cout << A + AA << endl;
	}

	cout << "############################ Constructors   ###########################" << endl << endl;

	cv::Vec3f v(1.0f, 0.0f, 2.0f);
	static const SparseMatrixCV B( v );
	cout << B << endl;


	cout << "############################ Multiplications ###########################" << endl << endl;
	{
		/* [19,  0, 21, 21,  0,  0;
			12, 21,  0,  0,  0,  0;
			 0, 12, 16,  0,  0,  0;
			 0,  0,  0,  5, 21,  0;
			12, 12,  0,  0, 18,  1]; */
		double non_zero_value_A[13] ={19, 21, 21, 12, 21, 12, 16,  5, 21, 12, 12, 18, 1};
		int    col_index_A[13] =     { 0,  2,  3,  0,  1,  1,  2,  3,  4,  0,  1,  4, 5};
		int    row_pointer_A[6] =    { 0,          3,      5,      7,      9,        13};
		SparseMatrixCV A(5, 6, non_zero_value_A, col_index_A, row_pointer_A, 13);


		/*[ 0, 14,  0,  0,  0;
		   12,  0,  0,  0,  7;
			0,  0,  0,  0,  0;
			0,  0,  0,  5, 21;
		   12, 12,  0,  0, 18;
			0,  0,  0,  0,  1;]; */
		double non_zero_value_B[9] ={14, 12,  7,  5, 21, 12, 12, 18,  1};
		int    col_index_B[9]      ={ 1,  0,  4,  3,  4,  0,  1,  4,  4};
		int    row_pointer_B[6]    ={ 0,  1,    3,3,      5,          8};
		SparseMatrixCV B(6, 5, non_zero_value_B, col_index_B, row_pointer_B, 9);
		cout << B << endl;

		SparseMatrixCV AmulB = A * B;
		// Expeted Out put
		// 0    266    0  105  441
		// 252  168    0    0  147
		// 144    0    0    0   84
		// 252  252    0   25  483
		// 360  384    0    0  409
		cout << AmulB << endl;


		Mat_<double> DenseB = ( Mat_<double>(6, 5)<<
			0, 14,  0,  0,  0,
		   12,  0,  0,  0,  7,
			0,  0,  0,  0,  0,
			0,  0,  0,  5, 21,
		   12, 12,  0,  0, 18,
			0,  0,  0,  0,  1 );
		cout << DenseB << endl << endl;

		Mat_<double> AmulDenseB = A * DenseB;
		cout << A * DenseB << endl;
	}

	cout << "############################ Matrix.inv() ###########################" << endl << endl;
	{
		double non_zero_value_A[9] ={ 2,  1,  3,  2,  6,  8,  6,  8, 18};
		int    col_index_A[9]      ={ 0,  1,  2,  0,  1,  2,  0,  1,  2};
		int    row_pointer_A[4]    ={ 0,          3,          6,          9};
		SparseMatrixCV A(3, 3, non_zero_value_A, col_index_A, row_pointer_A, 9);

		Mat_<double> B = (Mat_<double>(3,1) << 1, 3, 5 );

		cv::Mat_<double> X;

		// solve( A, B, X, SparseMatrixCV::SUPERLU );
		//cout << "X = " << endl << X << endl << endl;
		//cout << "A * X = " << endl << A * X << endl << endl;

		solve( A, B, X, SparseMatrixCV::BICGSQ );
		cout << "X = " << endl << X << endl << endl;
		cout << "A * X = " << endl << A * X << endl << endl;

		Mat DenseX;
		Mat_<double> DenseA;
		A.convertTo( DenseA );

		cv::solve( DenseA, B, DenseX, CV_64F );
		cout << DenseX << endl;
		cout << "A * X = " << endl << DenseA * DenseX << endl << endl;

		return 0;
	}

	cout << "############################ The End (^_^) ##########################" << endl << endl;
	return 0;

	// The following code is borrowed from SUPER_LU user manu.
	// The result of super_lu is incorrect.
	{
		SuperMatrix A, L, U, B;
		double *a, *rhs;
		double s, u, p, e, r, l;
		int *asub, *xa;
		int *perm_r; /* row permutations from partial pivoting */

		int *perm_c; /* column permutation vector */
		int nrhs, info, i, m, n, nnz;
		superlu_options_t options;
		SuperLUStat_t stat;
		/* Initialize matrix A. */
		m = n = 5;
		nnz = 12;
		if ( !(a = doubleMalloc(nnz)) ) ABORT("Malloc fails for a[].");
		if ( !(asub = intMalloc(nnz)) ) ABORT("Malloc fails for asub[].");
		if ( !(xa = intMalloc(n+1)) ) ABORT("Malloc fails for xa[].");
		s = 19.0; u = 21.0; p = 16.0; e = 5.0; r = 18.0; l = 12.0;
		a[0] = s; a[1] = l; a[2] = l; a[3] = u; a[4] = l; a[5] = l;
		a[6] = u; a[7] = p; a[8] = u; a[9] = e; a[10]= u; a[11]= r;
		asub[0] = 0; asub[1] = 1; asub[2] = 4; asub[3] = 1;
		asub[4] = 2; asub[5] = 4; asub[6] = 0; asub[7] = 2;
		asub[8] = 0; asub[9] = 3; asub[10]= 3; asub[11]= 4;
		xa[0] = 0; xa[1] = 3; xa[2] = 6; xa[3] = 8; xa[4] = 10; xa[5] = 12;
		/* Create matrix A in the format expected by SuperLU. */
		dCreate_CompCol_Matrix(&A, m, n, nnz, a, asub, xa, SLU_NC, SLU_D, SLU_GE);
		/* Create right-hand side matrix B. */
		nrhs = 1;
		if ( !(rhs = doubleMalloc(m * nrhs)) ) ABORT("Malloc fails for rhs[].");
		for (i = 0; i < m; ++i) rhs[i] = 1.0;
		dCreate_Dense_Matrix(&B, m, nrhs, rhs, m, SLU_DN, SLU_D, SLU_GE);
		if ( !(perm_r = intMalloc(m)) ) ABORT("Malloc fails for perm_r[].");
		if ( !(perm_c = intMalloc(n)) ) ABORT("Malloc fails for perm_c[].");
		/* Set the default input options. */
		set_default_options(&options);
		//options.ColPerm = NATURAL;
		//ilu_set_default_options( &options ); //
		//options.ILU_DropTol = 1e-26;

		/* Initialize the statistics variables. */
		StatInit(&stat);
		/* Solve the linear system. */
		dgssv(&options, &A, perm_c, perm_r, &L, &U, &B, &stat, &info);
		dPrint_CompCol_Matrix("A", &A);
		dPrint_CompCol_Matrix("U", &U);
		dPrint_SuperNode_Matrix("L", &L);
		print_int_vec("\nperm_r", m, perm_r);

		cout << "info: " << info << endl;
		cout << B.nrow << "," << B.ncol << endl;
		for( int i=0; i<m; i++ ) {
			/* Expetected values:
			-0.0466
			0.1057
			0.0162
			0.0736
			0.0301*/
			cout << ((double*)((DNformat*)B.Store)->nzval)[i] << endl;
		}

		/*
		A = [19,  0,    21,  21,  0;
		0.63, 21, -13.26, -13.26, 0;
		0, 0.57, 23.58, 7.58, 0;
		0, 0, 0, 5, 21;
		0.63, 0.57, -0.24, -0.77, 34.20;]
		*/

		/* De-allocate storage */
		SUPERLU_FREE (rhs);
		SUPERLU_FREE (perm_r);
		SUPERLU_FREE (perm_c);
		Destroy_CompCol_Matrix(&A);
		Destroy_SuperMatrix_Store(&B);
		Destroy_SuperNode_Matrix(&L);
		Destroy_CompCol_Matrix(&U);
		StatFree(&stat);
	}
	return 0;
}

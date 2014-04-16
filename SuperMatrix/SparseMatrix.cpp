#define _CRT_SECURE_NO_WARNINGS
#include "SparseMatrix.h"
#include <iostream> 

using namespace std;

SparseMatrix::SparseMatrix( int num_rows, int num_cols, const double non_zero_value[], 
	const int col_index[], const int row_pointer[], int N )
{
	data = new SparseMatrixData( num_rows, num_cols, non_zero_value, col_index, row_pointer, N );
	rc = new RC(); 
}


SparseMatrix::~SparseMatrix(void)
{
	if( rc->Release()==0 ){
		delete data;
		delete rc;
	}
}

SparseMatrix::SparseMatrix( const SparseMatrix& matrix ){
	this->data = matrix.data;
	this->rc   = matrix.rc;
	this->rc->AddRef();
}

const SparseMatrix& SparseMatrix::operator=( const SparseMatrix& matrix ){
	this->data = matrix.data;
	this->rc   = matrix.rc;
	this->rc->AddRef();
	return *this;
}









// friend functions

void solve( const SparseMatrix& AAAA, const double* BBBB, double* XXXX )
{
	
/*
 * Purpose
 * =======
 * 
 * This is the small 5x5 example used in the Sections 2 and 3 of the 
 * Users' Guide to illustrate how to call a SuperLU routine, and the
 * matrix data structures used by SuperLU.
 *
 */
    SuperMatrix A, L, U, B;
    double   *a, *rhs;
    double   s, u, p, e, r, l;
    int      *asub, *xa;
    int      *perm_r; /* row permutations from partial pivoting */
    int      *perm_c; /* column permutation vector */
    int      nrhs, info, i, m, n, nnz, permc_spec;
    superlu_options_t options;
    SuperLUStat_t stat;

    /* Initialize matrix A. */
    m = n = 5;
    nnz = 12;
    if ( !(a = doubleMalloc(nnz)) ) ABORT("Malloc fails for a[].");
    if ( !(asub = intMalloc(nnz)) ) ABORT("Malloc fails for asub[].");
    if ( !(xa = intMalloc(n+1)) )   ABORT("Malloc fails for xa[].");
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
    options.ColPerm = NATURAL;

    /* Initialize the statistics variables. */
    StatInit(&stat);

    /* Solve the linear system. */
    dgssv(&options, &A, perm_c, perm_r, &L, &U, &B, &stat, &info);
    
    dPrint_CompCol_Matrix("A", &A);
    dPrint_CompCol_Matrix("U", &U);
    dPrint_SuperNode_Matrix("L", &L);
    print_int_vec("\nperm_r", m, perm_r);

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


ostream& operator<<( ostream& out, const SparseMatrix& m ){
	const int& N         = m.data->getRow()->nnz(); 
	const double* const nzval = m.data->getRow()->nzvel(); 
	const int* const colidx   = m.data->getRow()->colinx(); 
	const int* const rowptr   = m.data->getRow()->rowptr(); 
	
	int vi = 0; 
	for( int r=0; r<m.row(); r++ ) {
		for( int c=0; c<m.col(); c++ ) {
			cout.width( 4 ); 
			if( colidx[vi]==c && vi<rowptr[r+1] ) cout << nzval[vi++] << " "; 
			else cout << 0 << " ";
		}
		cout << endl; 
	}

	return out; 
}

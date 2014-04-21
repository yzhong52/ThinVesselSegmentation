#define _CRT_SECURE_NO_WARNINGS
#include "SparseMatrix.h"
#include <vector>
#include <iostream> 

using namespace std;

SparseMatrix::SparseMatrix( int num_rows, int num_cols ) {
	data = new SparseMatrixData( num_rows, num_cols, 0, 0, 0, 0 ); 
	rc = new RC(); 
}
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

const SparseMatrix SparseMatrix::clone(void) const{
	// deep copy of the data
	SparseMatrix m( 
		this->row(),
		this->col(),
		(const double*)(this->data->getRow()->nzvel()),
		(const int*)(this->data->getRow()->colinx()),
		(const int*)(this->data->getRow()->rowptr()),
		this->data->getRow()->nnz() );
	return m; 
}

SparseMatrix::SparseMatrix( const SparseMatrix& matrix ){
	this->data = matrix.data;
	this->rc   = matrix.rc;
	this->rc->AddRef();
}

const SparseMatrix& SparseMatrix::operator=( const SparseMatrix& matrix ){
	this->~SparseMatrix(); 
	this->data = matrix.data;
	this->rc   = matrix.rc;
	this->rc->AddRef();
	return *this;
}

const SparseMatrix SparseMatrix::t() const{
	SparseMatrix m = this->clone(); 
	m.data->transpose(); 
	return m; 
}


const SparseMatrix& SparseMatrix::operator*=( const double& value ){
	this->data->multiply( value ); 
	return (*this);
}

const SparseMatrix& SparseMatrix::operator/=( const double& value ){
	this->data->multiply( 1.0/value ); 
	return (*this);
}




// friend functions

void solve( const SparseMatrix& AAAA, const double* BBBB, double* XXXX )
{
	// TODO: 
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
	if( m.data->getRow()==NULL ){
		cout << "This is a zero matrix." << endl;
		return out; 
	} 

	const int& N              = m.data->getRow()->nnz(); 
	const double* const nzval = m.data->getRow()->nzvel(); 
	const int* const colidx   = m.data->getRow()->colinx(); 
	const int* const rowptr   = m.data->getRow()->rowptr(); 
	
	int vi = 0; 
	cout << "Size: " << m.row() << " x " << m.col() << endl; 
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


const SparseMatrix operator*( const SparseMatrix& m1, const SparseMatrix& m2 ){
	assert( m1.col()==m2.row() && "Matrix size does not match" ); 

	if( m1.data->getRow()==NULL || m2.data->getCol()==NULL ){
		// if either m1 or m2 is zero matrix, return a zero matrix
		return SparseMatrix( m1.row(), m2.col() ); 
	} 

	vector<double> res_nzval;
	vector<int> res_colidx;
	vector<int> res_rowptr;

	const double* const nzval1 = m1.data->getRow()->nzvel(); 
	const int* const colidx1   = m1.data->getRow()->colinx(); 
	const int* const rowptr1   = m1.data->getRow()->rowptr(); 
	
	const double* const nzval2 = m2.data->getCol()->nzvel(); 
	const int* const rowidx2   = m2.data->getCol()->rowinx(); 
	const int* const colptr2   = m2.data->getCol()->colptr(); 
	
	// store the result as row-order
	res_rowptr.push_back( 0 ); 
	for( int r=0; r < m1.row(); r++ ) {
		for( int c=0; c < m2.col(); c++ ) {
			int r1 = rowptr1[r];
			int c2 = colptr2[c]; 
			double v = 0.0; 
			while( r1!=rowptr1[r+1] && c2!=colptr2[c+1] ) {
				if( colidx1[r1]==rowidx2[c2] ) v += nzval1[r1++] * nzval2[c2++];
				else if( colidx1[r1]<rowidx2[c2] ) r1++; 
				else c2++; 
			}
			if( v!=0.0 ) { 
				res_nzval.push_back( v ); 
				res_colidx.push_back( c ); 
			}
		}
		res_rowptr.push_back( (int) res_nzval.size() ); 
	}

	// use (const T*) to force the constructor to make a deep copy of the data
	SparseMatrix res( m1.row(), m2.col(),
		(const double*) (&res_nzval[0]),
		(const int*)    (&res_colidx[0]),
		(const int*)    (&res_rowptr[0]), 
		(int) res_nzval.size() );

	return res; 
}

const SparseMatrix operator-( const SparseMatrix& m1, const SparseMatrix& m2 ){
	assert( m1.row()==m2.row() && m1.col()==m2.col() && "Matrix size does not match" ); 

	
	if( m1.data->getRow()==NULL && m2.data->getRow()==NULL ){
		// if both of the are zero, return a zero matrix
		return SparseMatrix( m1.row(), m2.col() ); 
	} else if( m1.data->getRow()==NULL ){
		SparseMatrix res = m2.clone();
		return res*=-1.0;
	} else if( m2.data->getRow()==NULL ){
		return m1.clone();
	}

	vector<double> res_nzval;
	vector<int> res_colidx;
	vector<int> res_rowptr;

	const double* const nzval1 = m1.data->getRow()->nzvel(); 
	const int* const colidx1   = m1.data->getRow()->colinx(); 
	const int* const rowptr1   = m1.data->getRow()->rowptr(); 
	
	const double* const nzval2 = m2.data->getRow()->nzvel(); 
	const int* const colidx2   = m2.data->getRow()->colinx(); 
	const int* const rowptr2   = m2.data->getRow()->rowptr(); 

	// store the result as row-order
	res_rowptr.push_back( 0 ); 
	for( int r=0; r < m1.row(); r++ ) {
		int i1 = rowptr1[r];
		int i2 = rowptr2[r]; 
		while( i1!=rowptr1[r+1] && i2!=rowptr2[r+1] ) {
			const int& c1 = colidx1[i1];
			const int& c2 = colidx2[i2];

			double v = 0; 
			int c = min(c1, c2);
			if( c1==c2 ) {
				v = nzval1[i1++] - nzval2[i2++];
				if( abs(v)<1e-35 ) continue; 
			} else if( c1 < c2 ) {
				v = nzval1[i1++]; 
			} else {
				v = -nzval2[i2++]; 
			}
			res_nzval.push_back( v ); 
			res_colidx.push_back( c ); 
		}
		res_rowptr.push_back( (int) res_nzval.size() ); 
	}

	// use (const T*) to force the constructor to make a deep copy of the data
	SparseMatrix res( m1.row(), m2.col(),
		(const double*) (&res_nzval[0]),
		(const int*)    (&res_colidx[0]),
		(const int*)    (&res_rowptr[0]), 
		(int) res_nzval.size() );

	return res; 
}



const SparseMatrix operator+( const SparseMatrix& m1, const SparseMatrix& m2 ){
	assert( m1.row()==m2.row() && m1.col()==m2.col() && "Matrix size does not match" ); 

	if( m1.data->getRow()==NULL && m2.data->getRow()==NULL ){
		// if both of the are zero, return a zero matrix
		return SparseMatrix( m1.row(), m2.col() ); 
	} else if( m1.data->getRow()==NULL ){
		return m2.clone();
	} else if( m2.data->getRow()==NULL ){
		return m1.clone();
	}

	vector<double> res_nzval;
	vector<int> res_colidx;
	vector<int> res_rowptr;

	const double* const nzval1 = m1.data->getRow()->nzvel(); 
	const int* const colidx1   = m1.data->getRow()->colinx(); 
	const int* const rowptr1   = m1.data->getRow()->rowptr(); 
	
	const double* const nzval2 = m2.data->getRow()->nzvel(); 
	const int* const colidx2   = m2.data->getRow()->colinx(); 
	const int* const rowptr2   = m2.data->getRow()->rowptr(); 

	// store the result as row-order
	res_rowptr.push_back( 0 ); 
	for( int r=0; r < m1.row(); r++ ) {
		int i1 = rowptr1[r];
		int i2 = rowptr2[r]; 
		while( i1!=rowptr1[r+1] && i2!=rowptr2[r+1] ) {
			if( colidx1[i1]==colidx2[i2] ) {
				double v = nzval1[i1] + nzval2[i2];
				if( abs(v)>1e-35 ) { 
					res_nzval.push_back( v ); 
					res_colidx.push_back( colidx1[i1] ); 
				}
				i1++; 
				i2++;
			} else if( colidx1[i1]<colidx2[i2] ) {
				res_nzval.push_back( nzval1[i1] ); 
				res_colidx.push_back( colidx1[i1] ); 
				i1++; 
			} else {
				res_nzval.push_back( nzval2[i2] ); 
				res_colidx.push_back( colidx2[i2] ); 
				i2++; 
			}
		}
		res_rowptr.push_back( (int) res_nzval.size() ); 
	}

	// use (const T*) to force the constructor to make a deep copy of the data
	SparseMatrix res( m1.row(), m2.col(),
		(const double*) (&res_nzval[0]),
		(const int*)    (&res_colidx[0]),
		(const int*)    (&res_rowptr[0]), 
		(int) res_nzval.size() );

	return res; 
}


const SparseMatrix operator/( const SparseMatrix& m1, const double& value ){
	SparseMatrix sm = m1.clone();
	return (sm /= value);
}


const SparseMatrix operator*( const SparseMatrix& m1, const double& value ){
	SparseMatrix sm = m1.clone();
	return (sm *= value);
}
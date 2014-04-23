#define _CRT_SECURE_NO_WARNINGS
#include "SparseMatrix.h"
#include <vector>
#include <iostream> 

using namespace std;

SparseMatrix::SparseMatrix( int num_rows, int num_cols ) {
	// create a zero-matrix
	data = new SparseMatrixData( num_rows, num_cols ); 
	rc = new RC(); 
}
SparseMatrix::SparseMatrix( int num_rows, int num_cols, const double non_zero_value[], 
	const int col_index[], const int row_pointer[], int N )
{
	data = new SparseMatrixData( num_rows, num_cols, non_zero_value, col_index, row_pointer, N );
	rc = new RC(); 
}

SparseMatrix::SparseMatrix( int num_rows, int num_cols, 
		const std::vector<double> non_zero_value, 
		const std::vector<int> col_index, 
		const std::vector<int> row_pointer )
{
	assert( non_zero_value.size()==col_index.size() && row_pointer.size()==num_rows+1 && "Data size is invalid. " );
	data = new SparseMatrixData( num_rows, num_cols, 
		(const double*) &non_zero_value[0], 
		(const int*)    &col_index[0], 
		(const int*)    &row_pointer[0], 
		(int) non_zero_value.size() );
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
	if( this->isZero() ) {
		return SparseMatrix( this->row(), this->col() );
	} else {
		SparseMatrix m( 
			this->row(),
			this->col(),
			(const double*)(this->data->getRow()->nzvel()),
			(const int*)   (this->data->getRow()->colinx()),
			(const int*)   (this->data->getRow()->rowptr()),
			this->data->getRow()->nnz() );
		return m; 
	}
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

bool SparseMatrix::updateData( int num_rows, int num_cols, 
		const std::vector<double> non_zero_value, 
		const std::vector<int> col_index, 
		const std::vector<int> row_pointer )
{
	if( non_zero_value.size()==0 ) {
		delete data; 
		data = new SparseMatrixData( num_rows, num_cols );
		return true; 
	}else {
		return this->updateData( num_rows, num_cols, 
			(const double*) &non_zero_value[0], 
			(const int*)    &col_index[0], 
			(const int*)    &row_pointer[0], 
			(int) non_zero_value.size() );
	}
}

bool SparseMatrix::updateData(  int num_rows, int num_cols, 
	const double non_zero_value[], 
	const int col_index[], 
	const int row_pointer[], 
	int N )
{
	if( rc->num()!=1 ){
		std::cout << "Unable to update data, there are more than one reference. " << std::endl;
		return false; 
	}

	delete data; 
	data = new SparseMatrixData( num_rows, num_cols, non_zero_value, col_index, row_pointer, N );
	return true;
}

bool SparseMatrix::updateData(  int num_rows, int num_cols, 
	double non_zero_value[], 
	int col_index[], 
	int row_pointer[], 
	int N )
{
	if( rc->num()!=1 ){
		std::cout << "Unable to update data, there are more than one reference. " << std::endl;
		return false; 
	}

	delete data; 
	data = new SparseMatrixData( num_rows, num_cols, non_zero_value, col_index, row_pointer, N );
	return true;
}


// friend functions

void solve( const SparseMatrix& AAAA, const double* BBBB, double* XXXX )
{
    SuperMatrix L, U;
    
	int m = AAAA.row(); 
	int n = AAAA.col(); 

    /* Create right-hand side matrix B. */
    int nrhs = 1;
	double *rhs;
    if ( !(rhs = doubleMalloc(m * nrhs)) ) ABORT("Malloc fails for rhs[].");
    memcpy( rhs, BBBB, sizeof(double) * m );
	SuperMatrix B; 
    dCreate_Dense_Matrix(&B, m, nrhs, rhs, m, SLU_DN, SLU_D, SLU_GE);

	int *perm_r; /* row permutations from partial pivoting */
    int *perm_c; /* column permutation vector */
    if ( !(perm_r = intMalloc(m)) ) ABORT("Malloc fails for perm_r[].");
    if ( !(perm_c = intMalloc(n)) ) ABORT("Malloc fails for perm_c[].");
	
    /* Set the default input options. */
	superlu_options_t options;
    set_default_options( &options );
    options.ColPerm = NATURAL;

    /* Initialize the statistics variables. */
	SuperLUStat_t stat;
    StatInit( &stat );

    /* Solve the linear system. */
	int info; 
	dgssv( &options, 
		const_cast<SuperMatrix *>(AAAA.data->getRow()->getSuperMatrix()),
		perm_c, perm_r, &L, &U, 
		&B, // INPUT and OUTPUT
		&stat, &info);
    
	memcpy( XXXX, ((DNformat*)B.Store)->nzval, sizeof(double) * m );

    /* De-allocate storage */
    SUPERLU_FREE( rhs );
    SUPERLU_FREE( perm_r );
    SUPERLU_FREE( perm_c );
    
    Destroy_SuperMatrix_Store(&B);
    Destroy_SuperNode_Matrix(&L);
    Destroy_CompCol_Matrix(&U);
    StatFree(&stat);
}


ostream& operator<<( ostream& out, const SparseMatrix& m ){
	cout << "Size: " << m.row() << " x " << m.col() << endl; 

	if( m.data->getRow()==NULL ){
		cout << "  ...This is a zero matrix." << endl;
		return out; 
	} 

	const int& N              = m.data->getRow()->nnz(); 
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
		while( i1<rowptr1[r+1] && i2<rowptr2[r+1] ) {
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
		while( i1<rowptr1[r+1] ) {
			res_nzval.push_back( nzval1[i1] ); 
			res_colidx.push_back( colidx1[i1] ); 
			i1++;
		}
		while( i2<rowptr2[r+1] ) {
			res_nzval.push_back( -nzval2[i2] ); 
			res_colidx.push_back( colidx2[i2] ); 
			i2++; 
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
		while( i1<rowptr1[r+1] ) {
			res_nzval.push_back( nzval1[i1] ); 
			res_colidx.push_back( colidx1[i1] ); 
			i1++;
		}
		while( i2<rowptr2[r+1] ) {
			res_nzval.push_back( nzval2[i2] ); 
			res_colidx.push_back( colidx2[i2] ); 
			i2++; 
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
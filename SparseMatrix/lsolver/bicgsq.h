// Emacs should recognise this header as -*- C++ -*-


#ifndef BICGSQ_BLAS_H
#define BICGSQ_BLAS_H

// ============================================================================
//
//  CGS nach Sonneveld
//     CGS, a fast Lanczos-type solver for nonsymmetric linear systems
//     SIAM J Sci Stat Comput 10, 36-52 (1989)
//
//  siehe auch:
//  Ashby, Manteuffel, Saylor
//     A taxononmy for conjugate gradient methods
//     SIAM J Numer Anal 27, 1542-1568 (1990)
//
//  oder:
//  Willy D"orfler:
//     Orthogonale Fehlermethoden
//
//                                                 ----------------------------
//                                                 Christian Badura, Mai 1998
//
// ============================================================================


template< class MATRIX > inline
int
bicgsq( unsigned N, const MATRIX &A,
	const double *b, double *x, double eps );


template< class MATRIX > inline
int
bicgsq( unsigned N, const MATRIX &A,
	const double *b, double *x, double eps, bool detailed );



template< class MATRIX, class PC_MATRIX > inline
int
bicgsq( unsigned N, const MATRIX &A, const PC_MATRIX &C,
	const double *b, double *x, double eps );


template< class MATRIX, class PC_MATRIX > inline
int
bicgsq( unsigned N, const MATRIX &A, const PC_MATRIX &C,
	const double *b, double *x, double eps, bool detailed );


// ============================================================================


#include <assert.h>
#include "cblas.h"

// ============================================================================


template< class MATRIX, class PC_MATRIX > inline
int
bicgsq( unsigned N, const MATRIX &A, const PC_MATRIX &C,
	const double *b, double *x, double eps, bool detailed  ) {
  double *rT = new double[N];
  double *d  = new double[N];
  double *h  = new double[N];
  double *u  = new double[N];
  double *Sd = new double[N];
  double *t  = new double[N];
  double *Saut = u;
  double *aux1 = t;
  double *aux2 = Sd;
  double rTh, rTSd, rTr, alpha, beta;
  int its=0;
  double err=eps*eps*ddot(N,b,1,b,1);
  // f"ur's Abbruchkriterium (*)  -- r enth"alt immer das Residuum r=Ax-b
  double *r = new double[N];

  mult(A,x,r);
  daxpy(N,-1.,b,1,r,1);
  mult(C,r,d);
  dcopy(N,d,1,h,1);
  dcopy(N,h,1,u,1);
  dcopy(N,u,1,rT,1);
  assert( ddot(N,rT,1,rT,1)>1e-40 );
  rTh=ddot(N,rT,1,h,1);
  rTr=ddot(N,r,1,r,1);
  while ( rTr>err ) {
    mult(A,d,aux1);
    mult(C,aux1,Sd);
    rTSd=ddot(N,rT,1,Sd,1);
    assert( fabs(rTSd)>1e-40 );
    alpha=rTh/rTSd;
    dcopy(N,u,1,t,1);
    daxpy(N,-alpha,Sd,1,t,1);
    dscal(N,alpha,u,1);
    daxpy(N,alpha,t,1,u,1);
    daxpy(N,-1.,u,1,x,1);
    mult(A,u,aux2);
    daxpy(N,-1.,aux2,1,r,1);
    mult(C,aux2,Saut);
    daxpy(N,-1.,Saut,1,h,1);
    beta=1./rTh; rTh=ddot(N,rT,1,h,1); beta*=rTh;
    dcopy(N,h,1,u,1);
    daxpy(N,beta,t,1,u,1);
    dscal(N,beta*beta,d,1);
    daxpy(N,beta,t,1,d,1);
    daxpy(N,1.,u,1,d,1);
    rTr=ddot(N,r,1,r,1);
    if ( detailed )
      cout<<"bicgsq "<<its<<"\t"<<sqrt(rTr)<<endl;
    ++its;
  }
  delete[] r;
  delete[] rT;
  delete[] d;
  delete[] h;
  delete[] u;
  delete[] Sd;
  delete[] t;
  return its;
}


template< class MATRIX, class PC_MATRIX > inline
int
bicgsq( unsigned N, const MATRIX &A, const PC_MATRIX &C,
	const double *b, double *x, double eps ) {
  return bicgsq(N,A,C,b,x,eps,false);
}

// ============================================================================


template< class MATRIX > inline
int
bicgsq( unsigned N, const MATRIX &A,
	const double *b, double *x, double eps, bool detailed  ) {
  double *rT = new double[N];
  double *d  = new double[N];
  double *h  = new double[N];
  double *u  = new double[N];
  double *Ad = new double[N];
  double *t  = new double[N];
  double *Au = Ad;
  double rTh, rTAd, rTr, alpha, beta;
  int its=0;
  double err=eps*eps*ddot(N,b,1,b,1);
  // f"ur's Abbruchkriterium (*)  -- r enth"alt immer das Residuum r=Ax-b
  double *r = new double[N];

  mult(A,x,r);
  daxpy(N,-1.,b,1,r,1);
  dcopy(N,r,1,d,1);
  dcopy(N,r,1,h,1);
  dcopy(N,r,1,u,1);
  dcopy(N,u,1,rT,1);
  assert( ddot(N,rT,1,rT,1)>1e-40 );
  rTh=ddot(N,rT,1,h,1);
  rTr=ddot(N,r,1,r,1);
  while ( rTr>err ) {
    mult(A,d,Ad);
    rTAd=ddot(N,rT,1,Ad,1);
    assert( fabs(rTAd)>1e-40 );
    alpha=rTh/rTAd;
    dcopy(N,u,1,t,1);
    daxpy(N,-alpha,Ad,1,t,1);
    dscal(N,alpha,u,1);
    daxpy(N,alpha,t,1,u,1);
    daxpy(N,-1.,u,1,x,1);
    mult(A,u,Au);
    daxpy(N,-1.,Au,1,r,1);
    daxpy(N,-1.,Au,1,h,1);
    beta=1./rTh; rTh=ddot(N,rT,1,h,1); beta*=rTh;
    dcopy(N,h,1,u,1);
    daxpy(N,beta,t,1,u,1);
    dscal(N,beta*beta,d,1);
    daxpy(N,beta,t,1,d,1);
    daxpy(N,1.,u,1,d,1);
    rTr=ddot(N,r,1,r,1);
    if ( detailed )
      cout<<"bicgsq "<<its<<"\t"<<sqrt(rTr)<<endl;
    ++its;
  }
  delete[] r;
  delete[] rT;
  delete[] d;
  delete[] h;
  delete[] u;
  delete[] Ad;
  delete[] t;
  return its;
}


template< class MATRIX > inline
int
bicgsq( unsigned N, const MATRIX &A,
	const double *b, double *x, double eps ) {
  return bicgsq(N,A,b,x,eps,false);
}

// ============================================================================


#endif // BICGSQ_BLAS_H

// Emacs should recognise this header as -*- C++ -*-

#ifndef CGHS_BLAS_H
#define CGHS_BLAS_H

// ============================================================================
//
//  CG nach Hestenes und Stiefel
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


// ohne Vorkonditionierer
template< class MATRIX >
inline int
cghs( unsigned N, const MATRIX &A, const double *b, double *x, double eps );

template< class MATRIX >
inline int
cghs( unsigned N, const MATRIX &A, const double *b, double *x, double eps,
      bool detailed );


// mit Vorkonditionierer
template< class MATRIX, class PC_MATRIX >
inline int
cghs( unsigned N, const MATRIX &A, const PC_MATRIX &C,
      const double *b, double *x, double eps );

template< class MATRIX, class PC_MATRIX >
inline int
cghs( unsigned N, const MATRIX &A, const PC_MATRIX &C,
      const double *b, double *x, double eps, bool detailed );


// ============================================================================


#include "cblas.h"



template< class MATRIX >
inline
int
cghs( unsigned N, const MATRIX &A, const double *b, double *x, double eps,
      bool detailed ) {
  if ( N==0 )
    return -1;
  double *g = new double[N];
  double *r = new double[N];
  double *p = new double[N];
  int its=0;
  double t, tau, sig, rho, gam;
  double err=eps*eps*ddot(N,b,1,b,1);
  
  mult(A,x,g);
  daxpy(N,-1.,b,1,g,1);
  dscal(N,-1.,g,1);
  dcopy(N,g,1,r,1);
  while ( ddot(N,g,1,g,1)>err ) {
    mult(A,r,p);
    rho=ddot(N,p,1,p,1);
    sig=ddot(N,r,1,p,1);
    tau=ddot(N,g,1,r,1);
    t=tau/sig;
    daxpy(N,t,r,1,x,1);
    daxpy(N,-t,p,1,g,1);
    gam=(t*t*rho-tau)/tau;
    dscal(N,gam,r,1);
    daxpy(N,1.,g,1,r,1);
    if ( detailed )
      cout<<"cghs "<<its<<"\t"<<dnrm2(N,g,1)<<endl;
    ++its;
  }
  delete[] g;
  delete[] r;
  delete[] p;
  return its;
}


template< class MATRIX > inline int
cghs( unsigned N, const MATRIX &A, const double *b, double *x, double eps ) {
  return cghs(N,A,b,x,eps,false);
}


// ============================================================================


template< class MATRIX, class PC_MATRIX >
inline
int
cghs( unsigned N, const MATRIX &A, const PC_MATRIX &C,
      const double *b, double *x, double eps, bool detailed ) {
  if ( N==0 )
    return 0;
  double *r = new double[N];
  double *d = new double[N];
  double *h = new double[N];
  double *Ad = h;
  int its=0;
  double rh, alpha, beta;
  double err=eps*eps*ddot(N,b,1,b,1);

  mult(A,x,r);
  daxpy(N,-1.,b,1,r,1);
  mult(C,r,d);
  dcopy(N,d,1,h,1);
  rh=ddot(N,r,1,h,1);
  while ( ddot(N,r,1,r,1)>err ) {
    mult(A,d,Ad);
    alpha=rh/ddot(N,d,1,Ad,1);
    daxpy(N,-alpha,d,1,x,1);
    daxpy(N,-alpha,Ad,1,r,1);
    mult(C,r,h);
    beta=1./rh; rh=ddot(N,r,1,h,1); beta*=rh;
    dscal(N,beta,d,1);
    daxpy(N,1.,h,1,d,1);
    if ( detailed )
      cout<<"cghs "<<its<<"\t"<<dnrm2(N,r,1)<<endl;
    ++its;
  }
  delete[] r;
  delete[] d;
  delete[] h;
  return its;
}

template< class MATRIX, class PC_MATRIX > inline int
cghs( unsigned N, const MATRIX &A, const PC_MATRIX &C,
      const double *b, double *x, double eps ) {
  return cghs(N,A,C,b,x,eps,false);
}

// ============================================================================


#endif // CGHS_BLAS_H

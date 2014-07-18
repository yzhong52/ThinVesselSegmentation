#include "LevenbergMarquardt.h"

#include <iostream>
#include <iomanip>
#include <limits>
#include <omp.h>
#include <array>
#include <vector>

#include <sys/types.h> // For serialization
#include <sys/stat.h>  // For serialization
#include <unistd.h>    // For serialization

#include "Line3D.h"
#include "Neighbour26.h"
#include "Data3D.h"
#include "Timer.h"
#include "ModelSet.h"
#include "EnergyFunctions.h"
#include "SparseMatrixCV/SparseMatrixCV.h"


#if _MSC_VER && !__INTEL_COMPILER
#include <Windows.h>
#endif


using namespace std;
using namespace cv;


LevenbergMarquardt::LevenbergMarquardt( const vector<Vec3i>& dataPoints,
                                        const vector<int>& labelings,
                                        const ModelSet& modelset,
                                        const Data3D<int>& labelIDs,
                                        SmoothCostType smooth_cost_type )
    : tildaP( dataPoints ), labelID( labelings ), labelID3d( labelIDs )
    , modelset( modelset )
    , lines( modelset.lines )
{
    smart_assert( lines.size()!=0, "Error: model set is empty" );

    numParamPerLine = lines[0]->getNumOfParameters();
    numParam = numParamPerLine * (int) lines.size();

    // projection points of original points
    P = vector<Vec3d>( tildaP.size() );
    // Jacobian matrix of the projection points
    nablaP = vector<SparseMatrixCV>( tildaP.size() );

    using_smoothcost_func = nullptr;
}


void  LevenbergMarquardt::Jacobian_projection(
    const Vec3d& X1, const Vec3d& X2,
    const SparseMatrixCV& nablaX1, const SparseMatrixCV& nablaX2,
    const Vec3d& tildeP,           const SparseMatrixCV& nablaTildeP,
    Vec3d& P, SparseMatrixCV& nablaP )
{
    // Assume that the projection point P = T * X1 + (1-T) * X2
    Vec3d X1_X2 = X1 - X2;
    const double A = ( tildeP - X2 ).dot( X1_X2 );
    const double B = ( X1_X2 ).dot( X1_X2 );
    const double T = A / B;

    // Compute the Jacobian matrix for A and B
    const SparseMatrixCV nablaX1_nablaX2 = nablaX1 - nablaX2;
    const SparseMatrixCV nablaA = X1_X2.t() * SparseMatrixCV( nablaTildeP - nablaX2 ) + (tildeP - X2).t() * nablaX1_nablaX2;
    const SparseMatrixCV nablaB = X1_X2.t() * 2 * nablaX1_nablaX2;

    // Compute the Jacobian matrix for T (1 by N matrix)
    const SparseMatrixCV nablaT = ( nablaA * B - nablaB * A ) / ( B * B );

    // Compute the projection point (3 by 1 vector)
    P = T * X1 + (1-T) * X2;

    // And the Jacobian matrix of the projection point (3 by N matrix)
    nablaP = X1 * nablaT + nablaX1 * T + nablaX2 * (1-T) - X2 * nablaT;
}


SparseMatrixCV LevenbergMarquardt::Jacobian_datacost_for_one( const int& site )
{
    const int label = labelID[site];
    const Line3D* line = lines[label];

    // end points of the line
    Vec3d X1, X2;
    line->getEndPoints( X1, X2 );

    // Jacobian matrix of the end points of the line
    const unsigned indecesM1[][2] =
    {
        {0, 0 + label * numParamPerLine},
        {1, 1 + label * numParamPerLine},
        {2, 2 + label * numParamPerLine}
    };
    const unsigned indecesM2[][2] =
    {
        {0, 3 + label * numParamPerLine},
        {1, 4 + label * numParamPerLine},
        {2, 5 + label * numParamPerLine}
    };
    static const double values[] = { 1.0, 1.0, 1.0 };
    const SparseMatrixCV nablaX1( 3, numParam, indecesM1, values, 3 );
    const SparseMatrixCV nablaX2( 3, numParam, indecesM2, values, 3 );

    static const SparseMatrixCV zero_matrix(3, numParam);
    Jacobian_projection( X1, X2, nablaX1, nablaX2,
                         tildaP[site], zero_matrix,
                         P[site], nablaP[site] );

    const Vec3d tildeP_P = Vec3d(tildaP[site]) - P[site];
    const double tildaP_P_lenght = max( 1e-20, sqrt( tildeP_P.dot(tildeP_P) ) );

    // TODO: there is memory leak without clone()!
    return ( -1.0/tildaP_P_lenght*DATA_COST ) * tildeP_P.t() * nablaP[site].clone();
}



void LevenbergMarquardt::Jacobian_smoothcost_quadratic(
    const int& sitei, const int& sitej,
    SparseMatrixCV& nabla_smooth_cost_i,
    SparseMatrixCV& nabla_smooth_cost_j, void* func_data  )
{
    const int labeli = labelID[sitei];
    const int labelj = labelID[sitej];
    const Line3D* linei = lines[labeli];
    const Line3D* linej = lines[labelj];

    // end points of the lines
    Vec3d Xi1, Xi2, Xj1, Xj2;
    linei->getEndPoints( Xi1, Xi2 );
    linej->getEndPoints( Xj1, Xj2 );

    // Jacobian matrix of the end points of the line
    const unsigned indecesXi1[][2] =
    {
        {0, 0 + labeli * numParamPerLine},
        {1, 1 + labeli * numParamPerLine},
        {2, 2 + labeli * numParamPerLine}
    };
    const unsigned indecesXi2[][2] =
    {
        {0, 3 + labeli * numParamPerLine},
        {1, 4 + labeli * numParamPerLine},
        {2, 5 + labeli * numParamPerLine}
    };
    const unsigned indecesXj1[][2] =
    {
        {0, 0 + labelj * numParamPerLine},
        {1, 1 + labelj * numParamPerLine},
        {2, 2 + labelj * numParamPerLine}
    };
    const unsigned indecesXj2[][2] =
    {
        {0, 3 + labelj * numParamPerLine},
        {1, 4 + labelj * numParamPerLine},
        {2, 5 + labelj * numParamPerLine}
    };
    static const double values[] = { 1.0, 1.0, 1.0 };
    const SparseMatrixCV nablaXi1(3, numParam, indecesXi1, values, 3 );
    const SparseMatrixCV nablaXi2(3, numParam, indecesXi2, values, 3 );
    const SparseMatrixCV nablaXj1(3, numParam, indecesXj1, values, 3 );
    const SparseMatrixCV nablaXj2(3, numParam, indecesXj2, values, 3 );

    const Vec3d& Pi = P[sitei];
    const Vec3d& Pj = P[sitej];
    const SparseMatrixCV& nablaPi = nablaP[sitei];
    const SparseMatrixCV& nablaPj = nablaP[sitej];

    Vec3d Pi_prime, Pj_prime;
    SparseMatrixCV nablaPi_prime, nablaPj_prime;
    Jacobian_projection( Xj1, Xj2, nablaXj1, nablaXj2, Pi,      nablaPi,             Pi_prime, nablaPi_prime );
    Jacobian_projection( Xi1, Xi2, nablaXi1, nablaXi2, Pj,      nablaPj,             Pj_prime, nablaPj_prime );

    const double dist_pi_pj2       = max(1e-27, double( (Pi-Pj).dot(Pi-Pj) ) );
    const double dist_pi_pi_prime2 = max(1e-27, double( (Pi-Pi_prime).dot(Pi-Pi_prime) ) );
    const double dist_pj_pj_prime2 = max(1e-27, double( (Pj-Pj_prime).dot(Pj-Pj_prime))  );
    const double dist_pi_pj        = sqrt( dist_pi_pj2 );
    const double dist_pi_pi_prime  = sqrt( dist_pi_pi_prime2 );
    const double dist_pj_pj_prime  = sqrt( dist_pj_pj_prime2 );

    const SparseMatrixCV nabla_pi_pi_prime = ( Pi - Pi_prime ).t() * ( nablaPi - nablaPi_prime ) / dist_pi_pi_prime;
    const SparseMatrixCV nabla_pj_pj_prime = ( Pj - Pj_prime ).t() * ( nablaPj - nablaPj_prime ) / dist_pj_pj_prime;
    const SparseMatrixCV nabla_pi_pj       = ( Pi - Pj ).t() * ( nablaPi - nablaPj ) / dist_pi_pj;

    // output result
    nabla_smooth_cost_i = ( nabla_pi_pi_prime * dist_pi_pj - nabla_pi_pj * dist_pi_pi_prime ) * (1.0 / dist_pi_pj2 * PAIRWISE_SMOOTH);
    nabla_smooth_cost_j = ( nabla_pj_pj_prime * dist_pi_pj - nabla_pi_pj * dist_pj_pj_prime ) * (1.0 / dist_pi_pj2 * PAIRWISE_SMOOTH);
}


void LevenbergMarquardt::Jacobian_smoothcost_abs_esp( const int& sitei, const int& sitej,
        SparseMatrixCV& nabla_smooth_cost_i,
        SparseMatrixCV& nabla_smooth_cost_j, void* func_data )
{
    const int labeli = labelID[sitei];
    const int labelj = labelID[sitej];
    const Line3D* linei = lines[labeli];
    const Line3D* linej = lines[labelj];

    // end points of the lines
    Vec3d Xi1, Xi2, Xj1, Xj2;
    linei->getEndPoints( Xi1, Xi2 );
    linej->getEndPoints( Xj1, Xj2 );
//
//	// original points
//	const Vec3d tildePi = tildaP[sitei];
//	const Vec3d tildePj = tildaP[sitej];

    // Jacobian matrix of the end points of the line
    const unsigned indecesXi1[][2] =
    {
        {0, 0 + labeli * numParamPerLine},
        {1, 1 + labeli * numParamPerLine},
        {2, 2 + labeli * numParamPerLine}
    };
    const unsigned indecesXi2[][2] =
    {
        {0, 3 + labeli * numParamPerLine},
        {1, 4 + labeli * numParamPerLine},
        {2, 5 + labeli * numParamPerLine}
    };
    const unsigned indecesXj1[][2] =
    {
        {0, 0 + labelj * numParamPerLine},
        {1, 1 + labelj * numParamPerLine},
        {2, 2 + labelj * numParamPerLine}
    };
    const unsigned indecesXj2[][2] =
    {
        {0, 3 + labelj * numParamPerLine},
        {1, 4 + labelj * numParamPerLine},
        {2, 5 + labelj * numParamPerLine}
    };
    static const double values[] = { 1.0, 1.0, 1.0 };
    const SparseMatrixCV nablaXi1(3, numParam, indecesXi1, values, 3 );
    const SparseMatrixCV nablaXi2(3, numParam, indecesXi2, values, 3 );
    const SparseMatrixCV nablaXj1(3, numParam, indecesXj1, values, 3 );
    const SparseMatrixCV nablaXj2(3, numParam, indecesXj2, values, 3 );

    const Vec3d& Pi = P[sitei];
    const Vec3d& Pj = P[sitej];
    const SparseMatrixCV& nablaPi = nablaP[sitei];
    const SparseMatrixCV& nablaPj = nablaP[sitej];

    Vec3d Pi_prime, Pj_prime;
    SparseMatrixCV nablaPi_prime, nablaPj_prime;
    Jacobian_projection( Xj1, Xj2, nablaXj1, nablaXj2, Pi,      nablaPi,             Pi_prime, nablaPi_prime );
    Jacobian_projection( Xi1, Xi2, nablaXi1, nablaXi2, Pj,      nablaPj,             Pj_prime, nablaPj_prime );

    const double dist_pi_pj2       = max(1e-27, double( (Pi-Pj).dot(Pi-Pj) ) );
    const double dist_pi_pi_prime2 = max(1e-27, double( (Pi-Pi_prime).dot(Pi-Pi_prime) ) );
    const double dist_pj_pj_prime2 = max(1e-27, double( (Pj-Pj_prime).dot(Pj-Pj_prime))  );
    const double dist_pi_pj        = sqrt( dist_pi_pj2 );
    const double dist_pi_pi_prime  = sqrt( dist_pi_pi_prime2 );
    const double dist_pj_pj_prime  = sqrt( dist_pj_pj_prime2 );

    const SparseMatrixCV nabla_pi_pi_prime = ( Pi - Pi_prime ).t() * ( nablaPi - nablaPi_prime ) / dist_pi_pi_prime;
    const SparseMatrixCV nabla_pj_pj_prime = ( Pj - Pj_prime ).t() * ( nablaPj - nablaPj_prime ) / dist_pj_pj_prime;
    const SparseMatrixCV nabla_pi_pj       = ( Pi - Pj ).t() * ( nablaPi - nablaPj ) / dist_pi_pj;

    // output result
    std::pair<double,double>& oldsmoothcost = *((std::pair<double,double>*)func_data);
    nabla_smooth_cost_i = ( nabla_pi_pi_prime * dist_pi_pj - nabla_pi_pj * dist_pi_pi_prime )
                          * (1.0 / dist_pi_pj2 * PAIRWISE_SMOOTH * oldsmoothcost.first );
    nabla_smooth_cost_j = ( nabla_pj_pj_prime * dist_pi_pj - nabla_pi_pj * dist_pj_pj_prime )
                          * (1.0 / dist_pi_pj2 * PAIRWISE_SMOOTH * oldsmoothcost.second );
//	double temp = PAIRWISE_SMOOTH * oldsmoothcost.first;
}

void LevenbergMarquardt::Jacobian_datacost_thread_func(
    vector<double>& Jacobian_nzv,
    vector<unsigned>&    Jacobian_colindx,
    vector<unsigned>&    Jacobian_rowptr,
    vector<double>& energy_matrix,
    const int site )
{
    // For each data point, the following computation could
    // be splited into multiple thread
    const int& label = labelID[site];

    // computing datacost
    const double datacost_i = compute_datacost_for_one( lines[label], tildaP[site] );

    energy_matrix.push_back( sqrt(datacost_i) );

    // Computing derivative for data cost analytically
    const SparseMatrixCV J_datacost = Jacobian_datacost_for_one( site );

    unsigned nnz;
    const double* non_zero_value = NULL;
    const unsigned* column_index = NULL;
    const unsigned* row_pointer = NULL;
    J_datacost.getRowMatrixData( nnz, non_zero_value, column_index, row_pointer );

    smart_assert( J_datacost.row()==1 && J_datacost.col()==numParam,
                  "Number of row is not correct for Jacobian matrix" );

    for( unsigned i=0; i<nnz; i++ )
    {
        Jacobian_nzv.push_back( non_zero_value[i] );
        Jacobian_colindx.push_back( column_index[i] );
    }
    Jacobian_rowptr.push_back( (unsigned) Jacobian_nzv.size() );
}


void LevenbergMarquardt::Jacobian_datacosts_openmp(
    vector<double>& Jacobian_nzv,
    vector<unsigned>&    Jacobian_colindx,
    vector<unsigned>&    Jacobian_rowptr,
    vector<double>& energy_matrix )
{
    #pragma omp parallel /* Fork a team of threads*/
    {
        // local variables for different processes
        vector<double> Jacobian_nzv_loc;
        vector<unsigned>    Jacobian_colindx_loc;
        vector<unsigned>    Jacobian_rowptr_loc;
        vector<double> energy_matrix_loc;

        #pragma omp for

        for( int site = 0; site < (int) tildaP.size(); site++ )
        {
            // For each data point, the following computation will
            // be splited into multiple thread
            Jacobian_datacost_thread_func(
                Jacobian_nzv_loc,
                Jacobian_colindx_loc,
                Jacobian_rowptr_loc,
                energy_matrix_loc, site );
        }

        #pragma omp critical
        {
            Jacobian_nzv.insert(
                Jacobian_nzv.end(),
                Jacobian_nzv_loc.begin(),
                Jacobian_nzv_loc.end() );
            Jacobian_colindx.insert(
                Jacobian_colindx.end(),
                Jacobian_colindx_loc.begin(),
                Jacobian_colindx_loc.end());

            unsigned offset = Jacobian_rowptr.back();
            for( unsigned i=0; i<Jacobian_rowptr_loc.size(); i++ )
            {
                Jacobian_rowptr.push_back( Jacobian_rowptr_loc[i] + offset );
            }

            energy_matrix.insert(
                energy_matrix.end(),
                energy_matrix_loc.begin(),
                energy_matrix_loc.end() );
        }
    }
}

void LevenbergMarquardt::Jacobian_datacosts(
    vector<double>& Jacobian_nzv,
    vector<unsigned>&    Jacobian_colindx,
    vector<unsigned>&    Jacobian_rowptr,
    vector<double>& energy_matrix )
{
    for( unsigned site=0; site < tildaP.size(); site++ )
    {
        Jacobian_datacost_thread_func( Jacobian_nzv, Jacobian_colindx,
                                       Jacobian_rowptr, energy_matrix, site );
    }
}



void LevenbergMarquardt::Jacobian_smoothcost_thread_func(
    vector<double>& Jacobian_nzv,
    vector<unsigned>&    Jacobian_colindx,
    vector<unsigned>&    Jacobian_rowptr,
    vector<double>& energy_matrix,
    const int site )
{
    for( unsigned neibourIndex=0; neibourIndex<13; neibourIndex++ )   // find it's neighbour
    {
        Vec3i neig;  // the neighbour position
        Neighbour26::getNeigbour( neibourIndex, tildaP[site], neig );
        if( !labelID3d.isValid(neig) ) continue; // not a valid position

        const int site2 = labelID3d.at(neig);
        if( site2==-1 ) continue ; // not a neighbour

        const int l1 = labelID[site];
        const int l2 = labelID[site2];
        if( l1==l2 ) continue; // TODO

        double smoothcost_i_before = 0, smoothcost_j_before = 0;
        std::pair<double, double> coefficiency;
        using_smoothcost_func( lines[l1], lines[l2],
                               tildaP[site], tildaP[site2],
                               smoothcost_i_before, smoothcost_j_before, &coefficiency );

        // add more rows to energy_matrix according to smooth cost
        energy_matrix.push_back( sqrt( smoothcost_i_before ) );
        energy_matrix.push_back( sqrt( smoothcost_j_before ) );

        ////// Computing derivative of pair-wise smooth cost analytically
        SparseMatrixCV J[2];
        (this->*using_Jacobian_smoothcost_for_pair)( site, site2, J[0], J[1], &coefficiency );


        for( unsigned ji = 0; ji<2; ji++ )
        {
            unsigned nnz;
            const double* non_zero_value = NULL;
            const unsigned* column_index = NULL;
            const unsigned* row_pointer = NULL;
            J[ji].getRowMatrixData( nnz, non_zero_value, column_index, row_pointer );

            smart_assert( J[ji].row()==1 && J[ji].col()==numParam,
                          "Number of row is not correct for Jacobian matrix" );

            for( unsigned n1=0; n1 < nnz; n1++ )
            {
                Jacobian_nzv.push_back( non_zero_value[n1] );
                Jacobian_colindx.push_back( column_index[n1] );
            }
            Jacobian_rowptr.push_back( (int) Jacobian_nzv.size() );
        }

    } // end of - for each pair of pi and pj
}



void LevenbergMarquardt::Jacobian_smoothcosts_openmp(
    vector<double>& Jacobian_nzv,
    vector<unsigned>&    Jacobian_colindx,
    vector<unsigned>&    Jacobian_rowptr,
    vector<double>&	energy_matrix )
{
    // // // // // // // // // // // // // // // // // //
    // Construct Jacobian Matrix - smooth cost
    // // // // // // // // // // // // // // // // // //
//	const vector<Line3D*>& lines = modelset.models;
//	int numParamPerLine = lines[0]->getNumOfParameters();
//	const Data3D<int>& indeces = labelID3d;

    int max_num_threads = omp_get_max_threads(); // maximum number of thread
    vector<unsigned int> nzv_size( max_num_threads, 0);
    vector<unsigned int> nzv_rows( max_num_threads, 0);
    vector<unsigned int> accumulate_nzv_size( max_num_threads, 0);
    vector<unsigned int> accumulate_nzv_rows( max_num_threads, 0);

    #pragma omp parallel /* Fork a team of threads*/
    {
        // local variables for different processes
        vector<double> Jacobian_nzv_loc;
        vector<unsigned>    Jacobian_colindx_loc;
        vector<unsigned>    Jacobian_rowptr_loc;
        vector<double> energy_matrix_loc;
        #pragma omp for
        for( int site = 0; site < (int) tildaP.size(); site++ )   // For each data point
        {
            Jacobian_smoothcost_thread_func( Jacobian_nzv_loc, Jacobian_colindx_loc,
                                             Jacobian_rowptr_loc, energy_matrix_loc, site );
        } // end of contruction of Jacobian Matrix

        // obtatin current thread id
        int tid = omp_get_thread_num();
        // copy the data size of the current thead to global vectors nzv_size and nzv
        // 'global' here is with respect to 'non-local' for threads
        nzv_size[tid] = (unsigned int) Jacobian_nzv_loc.size();
        nzv_rows[tid] = (unsigned int) Jacobian_rowptr_loc.size();
        // original size of the global vectors
        unsigned int old_nzv_size = (unsigned int) Jacobian_nzv.size();
        unsigned int old_num_rows = (unsigned int) energy_matrix.size();

        smart_assert( Jacobian_nzv.size()==Jacobian_colindx.size(), "vector size mismatch. " );
        smart_assert( Jacobian_rowptr.size()-1==energy_matrix.size(), "vector size mismatch. " );

        #pragma omp barrier // wait for all thread to execute till this point

        #pragma omp single
        {
            int nThreads = omp_get_num_threads();
            accumulate_nzv_size[0] = nzv_size[0];
            accumulate_nzv_rows[0] = nzv_rows[0];
            for( int i=1; i<nThreads; i++ )
            {
                accumulate_nzv_size[i] = nzv_size[i] + accumulate_nzv_size[i-1];
                accumulate_nzv_rows[i] = nzv_rows[i] + accumulate_nzv_rows[i-1];
            }

            // resize the result vector, make them bigger!
            Jacobian_nzv.resize(     old_nzv_size + accumulate_nzv_size[nThreads-1] );
            Jacobian_colindx.resize( old_nzv_size + accumulate_nzv_size[nThreads-1] );
            Jacobian_rowptr.resize(  old_num_rows + accumulate_nzv_rows[nThreads-1]+1 );
            energy_matrix.resize(    old_num_rows + accumulate_nzv_rows[nThreads-1] );
        }

        #pragma omp barrier // wait for all thread to execute till this point

        // copy data to result vector
        int nzv_offset = old_nzv_size;
        if(tid>0) nzv_offset += accumulate_nzv_size[tid-1];

        // The following two memcpy is equivalent to the following for loop
        //for( unsigned int i=0; i<nzv_size[tid]; i++ ) {
        //	Jacobian_nzv[nzv_offset + i] = Jacobian_colindx_loc[i];
        //	Jacobian_colindx[nzv_offset+i] = Jacobian_colindx_loc[i];
        //}
        memcpy( &Jacobian_nzv[nzv_offset], &Jacobian_nzv_loc[0],
                nzv_size[tid] * sizeof(double) );
        memcpy( &Jacobian_colindx[nzv_offset], &Jacobian_colindx_loc[0],
                nzv_size[tid] * sizeof(int) );

        int row_offset = old_num_rows;
        if( tid>0 ) row_offset += accumulate_nzv_rows[tid-1];
        int data_offset = old_nzv_size;
        if( tid>0 ) data_offset += accumulate_nzv_size[tid-1];

        for( unsigned i=0; i<Jacobian_rowptr_loc.size(); i++ )
        {
            Jacobian_rowptr[ i + row_offset + 1 ] = Jacobian_rowptr_loc[i] + data_offset;
        }

        memcpy( &energy_matrix[row_offset], &energy_matrix_loc[0],
                nzv_rows[tid] * sizeof(double) );
    }
}

void LevenbergMarquardt::Jacobian_smoothcosts_openmp_critical_section(
    vector<double>& Jacobian_nzv,
    vector<unsigned>&    Jacobian_colindx,
    vector<unsigned>&    Jacobian_rowptr,
    vector<double>&	energy_matrix )
{
    #pragma omp parallel /* Fork a team of threads*/
    {
        // local variables for different processes
        vector<double> Jacobian_nzv_loc;
        vector<unsigned>    Jacobian_colindx_loc;
        vector<unsigned>    Jacobian_rowptr_loc;
        vector<double> energy_matrix_loc;

        #pragma omp for

        for( int site = 0; site < (int) tildaP.size(); site++ )
        {
            // For each data point, the following computation will
            // be splited into multiple thread
            Jacobian_smoothcost_thread_func(
                Jacobian_nzv_loc,
                Jacobian_colindx_loc,
                Jacobian_rowptr_loc,
                energy_matrix_loc, site );
        }

        #pragma omp critical
        {
            Jacobian_nzv.insert( Jacobian_nzv.end(),
            Jacobian_nzv_loc.begin(), Jacobian_nzv_loc.end());
            Jacobian_colindx.insert( Jacobian_colindx.end(),
            Jacobian_colindx_loc.begin(), Jacobian_colindx_loc.end());

            int offset = Jacobian_rowptr.back();
            for( unsigned i=0; i<Jacobian_rowptr_loc.size(); i++ )
            {
                Jacobian_rowptr.push_back( Jacobian_rowptr_loc[i] + offset );
            }

            energy_matrix.insert( energy_matrix.end(),
            energy_matrix_loc.begin(),
            energy_matrix_loc.end() );
        }
    }
}


void LevenbergMarquardt::Jacobian_smoothcosts(
    vector<double>& Jacobian_nzv,
    vector<unsigned>&    Jacobian_colindx,
    vector<unsigned>&    Jacobian_rowptr,
    vector<double>& energy_matrix )
{
    for( unsigned site = 0; site < tildaP.size(); site++ )
    {
        // For each data point, using only one thread for the computation
        Jacobian_smoothcost_thread_func( Jacobian_nzv, Jacobian_colindx, Jacobian_rowptr,
                                         energy_matrix, site );
    }
}






void LevenbergMarquardt::adjust_endpoints( void )
{
    // update the end points of the line
    vector<double> minT( (int) labelID.size(), (std::numeric_limits<double>::max)() );
    vector<double> maxT( (int) labelID.size(), (std::numeric_limits<double>::min)() );
    for( unsigned site = 0; site < tildaP.size(); site++ )   // For each data point
    {
        const int label = labelID[site];
        Vec3d p1, p2;
        lines[label]->getEndPoints( p1, p2 );

        const Vec3d& pos = p1;
        Vec3d dir = p2 - p1;
        dir /= sqrt( dir.dot( dir ) ); // normalize the direction
        double t = ( Vec3d(tildaP[site]) - pos ).dot( dir );
        maxT[label] = max( t+1, maxT[label] );
        minT[label] = min( t-1, minT[label] );
    }
    for( unsigned label=0; label < labelID.size(); label++ )
    {
        if( minT[label] < maxT[label] )
        {
            Vec3d p1, p2;
            lines[label]->getEndPoints( p1, p2 );

            const Vec3d& pos = p1;
            Vec3d dir = p2 - p1;
            dir /= sqrt( dir.dot( dir ) ); // normalize the direction

            lines[label]->setPositions( pos + dir * minT[label], pos + dir * maxT[label] );
        }
    }
}

void LevenbergMarquardt::update_lines( const Mat_<double>& delta )
{
    for( unsigned label=0; label < lines.size(); label++ )
    {
        for( unsigned i=0; i < numParamPerLine; i++ )
        {
            const double& d = delta.at<double>( label * numParamPerLine + i );
            lines[label]->updateParameterWithDelta( i, d );
        }
    }
}

void LevenbergMarquardt::reestimate( double lambda, SmoothCostType whatSmoothCost,
                                     const string& serialize_dataname )
{
    smart_assert( lines.size()!=0, "No line models available" );

    switch( whatSmoothCost )
    {
    case Linear:
        cout << endl << "Levenberg Marquardt::Linear" << endl;
        using_Jacobian_smoothcost_for_pair = &LevenbergMarquardt::Jacobian_smoothcost_abs_esp;
        using_smoothcost_func = &smoothcost_func_linear;
        break;
    case Quadratic:
        cout << endl << "Levenberg Marquardt::Quadratic" << endl;
        using_Jacobian_smoothcost_for_pair = &LevenbergMarquardt::Jacobian_smoothcost_quadratic;
        using_smoothcost_func = &smoothcost_func_quadratic;
        break;
    }

    cout << "Computing initial energy... ";
    cout.flush();
    double energy_before = compute_energy( tildaP, labelID, lines, labelID3d, using_smoothcost_func );
    cout << endl;

    // Identity matrix
    const SparseMatrixCV I  = SparseMatrixCV::I( numParam );

    // counting number in
    int energy_increase_count = 0;

    for( int lmiter = 0; lmiter<50; lmiter++ )
    {

        // Data for Jacobian matrix
        //  - # of cols: number of data points;
        //  - # of rows: number of parameters for all the line models
        vector<double> Jacobian_nzv;
        vector<unsigned> Jacobian_colindx;
        vector<unsigned> Jacobian_rowptr(1, 0);
        vector<double> energy_matrix;

        // // // // // // // // // // // // // // // // // //
        // Construct Jacobian Matrix -  data cost
        // // // // // // // // // // // // // // // // // //
        //Jacobian_datacosts( Jacobian_nzv, Jacobian_colindx, Jacobian_rowptr, energy_matrix );
        cout << "Compute data cost begin... ";
        cout.flush();
        Jacobian_datacosts_openmp( Jacobian_nzv, Jacobian_colindx, Jacobian_rowptr, energy_matrix );
        cout << "Done. " << endl;

        // // // // // // // // // // // // // // // // // //
        // Construct Jacobian Matrix - smooth cost
        // // // // // // // // // // // // // // // // // //
        cout << "Compute smooth cost begin... ";
        cout.flush();
        // Jacobian_smoothcosts_openmp_critical_section( Jacobian_nzv, Jacobian_colindx, Jacobian_rowptr, energy_matrix );
        Jacobian_smoothcosts_openmp( Jacobian_nzv, Jacobian_colindx, Jacobian_rowptr, energy_matrix );
        //Jacobian_smoothcosts( Jacobian_nzv, Jacobian_colindx, Jacobian_rowptr, energy_matrix );
        cout << "Done. " << endl;

        // Construct Jacobian matrix
        const SparseMatrixCV Jacobian = SparseMatrix(
                                            (int) Jacobian_rowptr.size() - 1,
                                            (int) lines.size() * numParamPerLine,
                                            Jacobian_nzv, Jacobian_colindx, Jacobian_rowptr );

        const SparseMatrixCV Jt = Jacobian.t();
        const SparseMatrixCV Jt_J = multiply_openmp( Jt, Jacobian );

        // const SparseMatrixCV A = Jt_J + Jt_J.diag() * lambda;
        const SparseMatrixCV A = Jt_J + I * lambda;

        // TODO: the following line could be optimized
        const Mat_<double> B = Jt * cv::Mat_<double>( (int) energy_matrix.size(), 1, &energy_matrix.front() ) ;

        Mat_<double> X;

        cout << "Solve linear equation begin... ";
        cout.flush();
        solve( A, B, X );
        cout << "Done. " << endl;


        update_lines( -X );

        cout << "Computing new energy... ";
        double new_energy = compute_energy( tildaP, labelID, lines, labelID3d, using_smoothcost_func );
        cout << "Done. " << endl;

        if( new_energy < energy_before )
        {
            // if energy is decreasing
            // adjust the endpoints of the lines
            adjust_endpoints();
            energy_before = new_energy;
            lambda *= 0.50;
            energy_increase_count = 0;
        }
        else
        {
            // if energy is increasing, reverse the result of this iteration
            update_lines( X );
            lambda *= 4.12;
            if( ++energy_increase_count>=3 )
            {
                // If energy_increase_count in three consecutive iterations
                // then the energy is probably converged
                break;
            }
        }

        cout << endl << " New Energy = "  << energy_before << endl;
        cout << " Lambda = " << lambda << endl << endl;

        cout << "Serialization begin. " << endl;
        modelset.serialize( serialize_dataname );
        cout << "Serialization done. " << endl  << endl;
    }
}




//for( int i=0; i<X.rows; i++ ) {
//	std::cout << std::setw(14) << std::scientific << X.at<double>(i) << "  ";
//}
//cout << endl;
//Sleep(500);


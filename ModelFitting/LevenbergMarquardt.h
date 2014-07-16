#pragma once
#include "SparseMatrixCV/SparseMatrixCV.h"
#include <opencv2/core/core.hpp>
#include "ModelSet.h"
#include "EnergyFunctions.h"
#include <array>
#include <vector>

using namespace cv;

class Line3D;
class ModelSet;
template <typename T> class Data3D;


extern const double DATA_COST;
extern const double PAIRWISE_SMOOTH;

class LevenbergMarquardt
{
public:
    enum SmoothCostType { Quadratic, Linear };

    LevenbergMarquardt( const vector<Vec3i>& dataPoints,
                        const vector<int>& labelings,
                        const ModelSet& modelset,
                        const Data3D<int>& labelIDs,
                        SmoothCostType smooth_cost_type = Quadratic );

    // lamda - damping function for levenburg marquart
    //    the smaller lambda is, the faster it converges
    //    the bigger lambda is, the slower it converges
    void reestimate( double lambda = 1e2,
                     SmoothCostType whatSmoothCost = Linear,
                     const string& serialize_dataname = "default" );

private:
    const vector<Vec3i>& tildaP;      /// Original positions of the points in 3D
    const vector<int>& labelID;       /// Corresponding labels of the points above

    const Data3D<int>& labelID3d;     /// Labeling of the data

    const ModelSet& modelset; /// A set of line models
    const vector<Line3D*>& lines;     // TODO: This is redundant

    unsigned numParamPerLine;
    unsigned numParam;

    vector<Vec3d>  P;              // projection points of original points
    vector<SparseMatrixCV> nablaP; // Jacobian matrix of the projection points

    SmoothCostFunc using_smoothcost_func;

private:
    // Jacobian Matrix - data cost
    void Jacobian_datacosts(
        vector<double>& Jacobian_nzv,
        vector<unsigned>&    Jacobian_colindx,
        vector<unsigned>&    Jacobian_rowptr,
        vector<double>& energy_matrix );
    // Jacobian Matrix - data cost
    void Jacobian_datacosts_openmp(
        vector<double>& Jacobian_nzv,
        vector<unsigned>&    Jacobian_colindx,
        vector<unsigned>&    Jacobian_rowptr,
        vector<double>& energy_matrix );
    // Jacobian Matrix - data cost - thread func
    void Jacobian_datacost_thread_func(
        vector<double>& Jacobian_nzv,
        vector<unsigned>&    Jacobian_colindx,
        vector<unsigned>&    Jacobian_rowptr,
        vector<double>& energy_matrix,
        const int site);

    // Jacobian Matrix - smooth cost
    void Jacobian_smoothcosts(
        vector<double>& Jacobian_nzv,
        vector<unsigned>&    Jacobian_colindx,
        vector<unsigned>&    Jacobian_rowptr,
        vector<double>& energy_matrix );
    // Jacobian Matrix - smooth cost
    void Jacobian_smoothcosts_openmp(
        vector<double>& Jacobian_nzv,
        vector<unsigned>&    Jacobian_colindx,
        vector<unsigned>&    Jacobian_rowptr,
        vector<double>& energy_matrix );
    // Jacobian Matrix - smooth cost
    void Jacobian_smoothcosts_openmp_critical_section(
        vector<double>& Jacobian_nzv,
        vector<unsigned>&    Jacobian_colindx,
        vector<unsigned>&    Jacobian_rowptr,
        vector<double>& energy_matrix );
    // Jacobian Matrix - smooth cost - thread func
    void Jacobian_smoothcost_thread_func(
        vector<double>& Jacobian_nzv,
        vector<unsigned>&    Jacobian_colindx,
        vector<unsigned>&    Jacobian_rowptr,
        vector<double>& energy_matrix,
        const int site );

private:
    // X1, X2: 3 by 1 vector, end point of the lines
    // nablaX1, nablaX2: 3 * numParam, 3 non-zero values, Jacobian matrix of the end points of the line
    // P: 3 by 1, OUTPUT, projection point
    // nablaP: 1 by N, OUTPUT, jacobian matrix of the projection point
    void  Jacobian_projection(
        const cv::Vec3d& X1, const cv::Vec3d& X2,                                    // two end points of a line
        const SparseMatrixCV& nablaX1, const SparseMatrixCV& nablaX2,          // Jacobians of the end points of the line
        const cv::Vec3d& tildeP,           const SparseMatrixCV& nablaTildeP,      // a point, and the Jacobian of the point
        cv::Vec3d& P, SparseMatrixCV& nablaP );

    SparseMatrixCV Jacobian_datacost_for_one( const int& site );

    void (LevenbergMarquardt::*using_Jacobian_smoothcost_for_pair)( const int& sitei, const int& sitej,
            SparseMatrixCV& nabla_smooth_cost_i,
            SparseMatrixCV& nabla_smooth_cost_j, void* func_data );

    // for two different type of smooth cost
    void Jacobian_smoothcost_abs_esp( const int& sitei, const int& sitej,
                                      SparseMatrixCV& nabla_smooth_cost_i,
                                      SparseMatrixCV& nabla_smooth_cost_j, void* func_data = NULL );
    void Jacobian_smoothcost_quadratic( const int& sitei, const int& sitej,
                                        SparseMatrixCV& nabla_smooth_cost_i,
                                        SparseMatrixCV& nabla_smooth_cost_j, void* func_data = NULL );

    // Update model according to delta
    // (delta can be consider as the gradient computed with levenberg marquart)
    void update_lines( const Mat_<double>& delta );

    // adjust the end points of the lines so that they don't shift away from the data
    void adjust_endpoints( void );
};


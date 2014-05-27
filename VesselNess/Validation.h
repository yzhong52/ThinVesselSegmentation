#pragma once

template<typename T> class Data3D;
template<typename T> class Image3D;

namespace Validation
{
namespace Eigenvalues
{
void plot_1d_box(void);
void plot_2d_tubes(void);
void plot_2d_ball(void);
void plot_3d_tubes(void);
}

// What model should we use for ball fitting?
namespace BallFittingModels
{
void cylinder(void);
void circular_truncated_cone(void);
void gaussian(void);
void laplacian_of_gaussian(void);
}

// plot 2n derivative of gaussian on 1d box function
namespace box_func_and_2nd_gaussian
{
void plot_different_size(void);
void plot_different_pos(void);
void plot(void);
}

// explore the difference between
// 1) Harris Detector
// 2) Hessian Matrix
// 3) Optimal Oriented Flux
bool Harris_Hessian_OOP(void);

bool Hessian_3D(void);
bool Hessian_2D(void);

bool Rings_Reduction_Polar_Coordinates( const Mat& im, Mat& dst, int wsize );
bool Rings_Reduction_Cartecian_Coordinates( const Mat& src, Mat& dst );

void construct_tube(  Data3D<short>& image );
void construct_tube2( Data3D<short>& image );
}


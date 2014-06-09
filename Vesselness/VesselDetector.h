#ifndef VESSELDETECTOR_H
#define VESSELDETECTOR_H

#include "VesselnessTypes.h"

template<typename T> class Data3D;
template<typename T> class Image3D;
template<typename T> class Kernel3D;

class Vesselness;
class Vesselness_Sig;
class Vesselness_Nor;
class Vesselness_All;

namespace VesselDetector
{
bool hessien(
    const Data3D<short>& src, Data3D<Vesselness>& dst,
    int ksize, float sigma,
    float alpha, float beta, float gamma );

int compute_vesselness(
    const Data3D<short>& src,                           // INPUT
    Data3D<Vesselness_Sig>& dst,                        // OUTPUT
    float sigma_from, float sigma_to, float sigma_step, // INPUT
    float alpha = 1.0e-1f,	                            // INPUT
    float beta  = 5.0e0f,	                            // INPUT
    float gamma = 3.5e5f );                             // INPUT
};

namespace VD = VesselDetector;
namespace VF = VesselDetector;


#endif // VESSELDETECTOR_H



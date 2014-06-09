#include "VesselnessTypes.h"
#include "ImageProcessing.h"

namespace ImageProcessing
{

void non_max_suppress( const Data3D<Vesselness_Sig>& src, Data3D<Vesselness_Sig>& dst );

void edge_tracing( Data3D<Vesselness_Sig>& src, Data3D<Vesselness_Sig>& dst, const float& thres1, const float& thres2 );

void dir_tracing( Data3D<Vesselness_All>& src_vn, Data3D<Vesselness_Sig>& dst );

void edge_tracing_mst( Data3D<Vesselness_All>& src_vn, Data3D<Vesselness_Sig>& dst, const float& thres1, const float& thres2  );

}

#pragma once

#include <vector>
#include <iostream>
#include <string>
#include <fstream>
#include <opencv2/core/core.hpp>

#include "VesselnessTypes.h"
#include "Data3D.h"

class Line3D;
class Vesselness_Sig;

class ModelSet
{
public:
    // a group of 3D data points to be fit into line models
    std::vector<cv::Vec3i> tildaP;

    // the corresponding labeling of the 3D data points
    // (same size of tildaP.size())
    std::vector<int> labelID;

    // line models
    std::vector<Line3D*> lines;

    // The indexing of the data points in the 3D volume
    // (for fast accessing under certain conditions)
    Data3D<int> labelID3d;

    inline int get_data_size();

    ModelSet(void);
    virtual ~ModelSet(void);

    /// Serialization & Deserialization
    void serialize( std::string file ) const;
    bool deserialize( std::string file );

    ////////////////////////////////////////////////////////////////
    // Model Initialization
    ////////////////////////////////////////////////////////////////
    void init_one_model_per_point( const Data3D<Vesselness_Sig>& vn_sig,
                                   const float& threshold = 0.1f );
private:
    static const std::string file_prefix;

};




inline int ModelSet::get_data_size(void)
{
    return tildaP.size();
}



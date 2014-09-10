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
    // A group of 3D data points to be fit into line models and their
    // corresponding labeling
    std::vector<cv::Vec3i> tildaP;
    std::vector<int> labelID;
    /*Todo: since the above two have the same size, it is better to put them
      into one big vector<std::pair<Vec3i, int> > or something similar. */

    // Line models
    std::vector<Line3D*> lines;

    /* The labeling of 3D data points. '-1' indicates that it is a background
       label. For example, to get the line of point 'p', do the following:
           if( labelID3d.at(p) != -1 ){
               Line3D* l = lines[ labelID3d.at(p) ];
           }
       This is used in Model Fitting.
       TODO: it is redundant with pointID3d, see if you can remove this.
    */
    Data3D<int> labelID3d;

    /* The position of the current point in tildaP vector. '-1' indicates
       that it is a background label. For example, to get the line of
       point 'p', do the following:
           if( pointID3d.at(p) != -1 ) {
               assert( tildaP[ pointID3d.at(p) ] == p );
               int lineid = lableID[ pointID3d.at(p) ];
               Line3D* l = lines[ lineid ];
           }
       */
    Data3D<int> pointID3d;

    inline int get_data_size();

    ModelSet(void);
    virtual ~ModelSet(void);

    /// Serialization & Deserialization
    void serialize( std::string file ) const;
    bool deserialize( std::string file );
    // A point 'p' is ignored if mask.at(p)!=0.
    bool deserialize( std::string file, const Data3D<unsigned char>& mask );

    ////////////////////////////////////////////////////////////////
    // Model Initialization
    ////////////////////////////////////////////////////////////////
    void init_one_model_per_point( const Data3D<Vesselness_Sig>& vn_sig,
                                   const float& threshold = 0.1f );

};




inline int ModelSet::get_data_size(void)
{
    return tildaP.size();
}



#pragma once
#include <vector>
#include <iostream>
#include <string>
#include <fstream>

#include "Line3DTwoPoint.h"
class Line3D;

class ModelSet
{
public:
    std::vector<Line3D*> models;

    ModelSet(void);
    virtual ~ModelSet(void);

    /// Serialization
    void serialize( std::string file ) const;
    /// Deserialization - Actual Type is for polymorphism
    void deserialize( std::string file );
};



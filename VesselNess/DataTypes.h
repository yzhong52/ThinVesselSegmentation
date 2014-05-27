#pragma once

#include "Edge.h"
#include <vector>
#include <algorithm>    // std::max

/////////////////////////////////////////////////////////////
// Some basic data structures
/////////////////////////////////////////////////////////////
namespace MinSpanTree
{
struct Vec3f
{
    float x, y, z;
    // constructor
    Vec3f(const Vec3f&v ) : x(v.x), y(v.y), z(v.z) { }
    Vec3f( float x=0, float y=0, float z=0 ) : x(x), y(y), z(z) { }

    // operator
    float dot( const Vec3f& v ) const
    {
        return x * v.x + y * v.y + z * v.z;
    }
    inline Vec3f operator-( const Vec3f& v) const
    {
        return Vec3f( x-v.x, y-v.y, z-v.z);
    }
    inline Vec3f operator+( const Vec3f& v) const
    {
        return Vec3f( x+v.x, y+v.y, z+v.z);
    }
    inline Vec3f operator*( const float& v) const
    {
        return Vec3f( v*x, v*y, v*z );
    }
    inline Vec3f cross( const Vec3f& v ) const
    {
        Vec3f res;
        res.x = y * v.z - z * v.y;
        res.y = z * v.x - x * v.z;
        res.z = x * v.y - y * v.x;
        return res;
    }
    inline float length() const
    {
        return sqrt(x*x + y*y + z*z);
    }
};

struct Point3D
{
    int x, y, z;
};

struct LineSegment
{
    Vec3f p1, p2;
    float radius;
    std::vector<Point3D> points; // the points that are assigned to this label
    LineSegment( )
        : p1( Vec3f(0,0,0) )
        , p2( Vec3f(0,0,0) )
        , radius(0.0f)
        , points( std::vector<Point3D>() )
    {

    }
    LineSegment(const LineSegment& l )
        : p1(l.p1)
        , p2(l.p2)
        , radius(l.radius)
        , points(l.points)
    {

    }
    void get_distance( const Vec3f& from,
                       /*Output*/ Vec3f& to,
                       /*Output*/ float& distance )
    {
        Vec3f temp1 = p2 - p1;
        Vec3f temp2 = p1 - from;
        float numerator = temp2.dot( temp1 );
        float denominator = temp1.dot( temp1 );
        float t = -numerator / denominator;
        // constrain t to be within 0 and 1
        if( t>=1.0f )
        {
            t = 1.0f;
        }
        else if( t< 0.0f )
        {
            t = 0.0f;
        }
        to = p2 * t + p1 * (1-t);
        Vec3f dir = from - to;
        distance = dir.dot( dir );
    }
};

struct Edge_Ext : public Edge
{
    LineSegment line;
};
}

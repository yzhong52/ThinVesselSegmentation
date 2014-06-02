#include "Line3DTwoPoint.h"

using namespace std;
using namespace cv;

Line3DTwoPoint::Line3DTwoPoint(void)
{
}


Line3DTwoPoint::~Line3DTwoPoint(void)
{
}


double Line3DTwoPoint::distanceToLine( const Vec3d& point ) const
{
    // get the porjection from the point to the line
    Vec3d proj_point = projection( point );
    Vec3d v = proj_point - point;
    return sqrt( v.dot(v) );
}

Vec3d Line3DTwoPoint::projection( const Vec3d& point ) const
{
    Vec3d pos = cv::Vec3d( &data[0] );
    Vec3d dir = cv::Vec3d( &data[3] ) - cv::Vec3d( &data[0] );
    dir /= sqrt( dir.dot( dir ) ); // normalize the direction
    double t = ( point-pos ).dot( dir );
    return pos + dir * t;
}

double Line3DTwoPoint::loglikelihood( const Vec3d& point ) const
{
    double dist = this->distanceToLine( point );
    return dist * dist / this->sigma;
}

void Line3DTwoPoint::updateParameterWithDelta( int i, double delta )
{
    data[i] += (double) delta;
}



Vec3d Line3DTwoPoint::getDirection( void ) const
{
    Vec3d dir = Vec3d( &data[0] ) - Vec3d( &data[3] );
    dir /= sqrt( dir.dot( dir ) );
    return dir;
}

void Line3DTwoPoint::getEndPoints( Vec3d& p1, Vec3d& p2 ) const
{
    p1 = Vec3d( &data[0] );
    p2 = Vec3d( &data[3] );
}

void Line3DTwoPoint::serialize( std::ostream& out ) const
{
    out << this->sigma << " ";
    for( int i=0; i<6; i++ ) out << this->data[i] << " ";
    out << endl;
}

void Line3DTwoPoint::deserialize( std::istream& in )
{
    in >> this->sigma;
    for( int i=0; i<6; i++ ) in >> data[i];
}

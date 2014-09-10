#pragma once

#include <vector>
#include <opencv2/core/core.hpp>
#include "smart_assert.h"

class Neighbour26
{
    cv::Vec3i offset[26];
    cv::Vec3d offset_f[26];
    std::vector<cv::Vec3i> cross_section[13];

    Neighbour26(void);
    virtual ~Neighbour26(void) { }

    inline static Neighbour26& getInstance();
public:

    inline static const cv::Vec3i& at( int index );
    inline static const std::vector<cv::Vec3i>& getCrossSection( int index );
    inline static void at( int index, int& x, int& y, int& z );

    inline static void getNeigbour( int index,
                                    const cv::Vec3i& pos,
                                    cv::Vec3i& neigbour );
    inline static void getNeigbour( int index,
                                    const int& old_x,
                                    const int& old_y,
                                    const int& old_z,
                                    int& x, int& y, int& z );
};

inline Neighbour26::Neighbour26(void)
{
    for( int i=0; i<26; i++ )
    {
        int index = (i + 14) % 27;
        offset[i][0] = index/9%3 - 1;
        offset[i][1] = index/3%3 - 1;
        offset[i][2] = index/1%3 - 1;

        // normalize offset_f
        offset_f[i] = cv::Vec3d( offset[i] );
        float length = offset_f[i].dot( offset_f[i] );
        length = sqrt( length );
        offset_f[i] /= length;
    }

    // cross section for the major orientation
    // Setting offsets that are perpendicular to dirs
    for( int i=0; i<13; i++ )
    {
        for( int j=0; j<26; j++ )
        {
            // multiply the two directions
            float temp = offset_f[j].dot( offset[i] );
            // the temp is 0, store this direciton
            if( abs(temp)<1.0e-5 )
            {
                cross_section[i].push_back( offset[j] );
            }
        }
    }
}


inline Neighbour26& Neighbour26::getInstance()
{
    static Neighbour26 instance;
    return instance;
}

inline const cv::Vec3i& Neighbour26::at( int index )
{
    smart_assert( index>=0 && index<26, "Invalid index" );
    return getInstance().offset[index];
}

inline const std::vector<cv::Vec3i>& Neighbour26::getCrossSection( int index )
{
    smart_assert( index>=0 && index<13, "Invalid index" );
    return getInstance().cross_section[ index ];
}


inline void Neighbour26::at( int index, int& x, int& y, int& z )
{
    if( index<0 || index>=26 )
    {
        std::cout << "Index Error for 26 neigbours" << std::endl;
        system("pause");
        x = y = z = 0;
        return;
    }
    x = getInstance().offset[index][0];
    y = getInstance().offset[index][1];
    z = getInstance().offset[index][2];
}

inline void Neighbour26::getNeigbour( int index, const cv::Vec3i& pos, cv::Vec3i& neigbour )
{
    return getNeigbour(index, pos[0], pos[1], pos[2], neigbour[0], neigbour[1], neigbour[2]);
}


inline void Neighbour26::getNeigbour( int index, const int& old_x, const int& old_y, const int& old_z,
                                      int& x, int& y, int& z )
{
    smart_assert( index>=0 && index<26, "Index Error for 26 neigbours" );

    x = getInstance().offset[index][0] + old_x;
    y = getInstance().offset[index][1] + old_y;
    z = getInstance().offset[index][2] + old_z;
}

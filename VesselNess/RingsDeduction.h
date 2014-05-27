#pragma once

template<typename T> class Data3D;

namespace RingsDeduction
{
void mm_filter( Data3D<short>& im, const int& wsize=15,
                const int& center_x = 234, const int& center_y = 270 );

void gm_filter( Data3D<short>& im, const int& wsize=21,
                const int& center_x = 234, const int& center_y = 270 );
};

namespace RD = RingsDeduction;

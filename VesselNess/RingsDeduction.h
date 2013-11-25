#pragma once

template<typename T> class Data3D;

namespace RingsDeduction
{
	void mm_filter( Data3D<short>& im, const int& wsize=15 );
};

namespace RD = RingsDeduction; 
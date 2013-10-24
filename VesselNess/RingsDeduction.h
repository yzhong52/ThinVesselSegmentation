#pragma once

template<typename T> class Image3D;

namespace RingsDeduction
{
	void mm_filter( Image3D<short>& im, const int& wsize=15 );
};


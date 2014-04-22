#pragma once

// A reference counting class RC. 
// This class maintains an integer value which represents the reference count.
// We will have methods to increment and decrement the reference count.
class RC
{
private:
	int count; // Reference count

public:
	// Initially, reference count is 1
	RC() : count(1) { }

	inline int num(void) const { return count; }

	// Increment the reference count
	void AddRef(){ count++; }

	// Decrement the reference count and
	// return the reference count.
	int Release(){ return --count; }
};

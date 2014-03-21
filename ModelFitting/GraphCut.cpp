#include "GraphCut.h"

#include <iostream> 
#include <iomanip>
#include <Windows.h>
using namespace std; 

#include "Line3D.h" 

// For the use of graph cut
#include "GCoptimization.h"

typedef GCoptimization GC; 


double GraphCut::estimation(  const vector<Vec3i>& dataPoints,
		vector<int>& labelings, 
		const vector<Line3D*>& lines  )
{
	GCoptimizationGeneralGraph gc( (int) dataPoints.size(), (int) lines.size() ); 
	////////////////////////////////////////////////
	// Data Costs
	////////////////////////////////////////////////
	for(int site = 0; site < (GC::SiteID) dataPoints.size(); site++ ) {
		for( int label = 0; label < lines.size(); label++ ){
			const Line3D* line = lines[label];
			GC::EnergyTermType loglikelihood = line->loglikelihood( dataPoints[site] ); 
			gc.setDataCost( site, label, LOGLIKELIHOOD * loglikelihood );
		}
	}

	////////////////////////////////////////////////
	// Smooth Cost
	////////////////////////////////////////////////
	// ... TODO: Setting Smooth Cost
	// im_uchar
	

	////////////////////////////////////////////////
	// Graph-Cut Begin
	////////////////////////////////////////////////
	// cout << "Iteration: " << i << ". Fitting Begin. Please Wait..."; 
	gc.expansion(1); // run expansion for 1 iterations. For swap use gc->swap(num_iterations);

	GC::EnergyType energy = gc.compute_energy();
	
	// update labelings
	for( GC::SiteID site = 0; site < (GC::SiteID) dataPoints.size(); site++ ) {
		const int& x = dataPoints[site][0];
		const int& y = dataPoints[site][1];
		const int& z = dataPoints[site][2];
		labelings[site] = gc.whatLabel( site ); 
	}

	return energy; 
}
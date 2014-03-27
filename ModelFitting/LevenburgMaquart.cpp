#include "LevenburgMaquart.h"

#include <iostream> 
#include <iomanip>
#include <Windows.h>
using namespace std; 

#include "Line3D.h" 
#include "Neighbour26.h"
#include "Data3D.h" 

inline double compute_energy_datacost_for_one( 
	const Line3D* line_i,
	const Vec3i& pi )
{
	return LOGLIKELIHOOD* line_i->loglikelihood( pi );
}

double compute_energy_datacost( 
	const vector<Vec3i>& dataPoints,
	const vector<int>& labelings, 
	const vector<Line3D*>& lines )
{
	double energy = 0; 
	for( int site = 0; site < (int) dataPoints.size(); site++ ) {
		int label = labelings[site];
		energy += compute_energy_datacost_for_one( lines[label], dataPoints[site] ); 
	}
	return energy; 
}


void compute_energy_smoothcost_for_pair( 
	const Line3D* line_i,
	const Line3D* line_j,
	const Vec3i& pi_tilde, 
	const Vec3i& pj_tilde,
	double smooth_cost_i, 
	double smooth_cost_j )
{
	// single projection
	Vec3f pi = line_i->projection( pi_tilde ); 
	Vec3f pj = line_j->projection( pj_tilde ); 

	// double projection
	Vec3f pi_prime = line_j->projection( pi ); 
	Vec3f pj_prime = line_i->projection( pj ); 

	// distance vector
	Vec3f pipj  = pi - pj; 
	Vec3f pipi1 = pi - pi_prime; 
	Vec3f pjpj1 = pj - pj_prime;

	// distance
	float dist_pipj  = pipj.dot(pipj); 
	float dist_pipi1 = pipi1.dot(pipi1); 
	float dist_pjpj1 = pjpj1.dot(pjpj1); 

	smooth_cost_i = dist_pipi1 / dist_pipj; 
	smooth_cost_j = dist_pjpj1 / dist_pipj; 
}

double compute_energy_smoothcost( 
	const vector<Vec3i>& dataPoints,
	const vector<int>& labelings, 
	const vector<Line3D*>& lines,
	const Data3D<int>& indeces )
{
	double energy = 0; 

	for( int site = 0; site < dataPoints.size(); site++ ) { // For each data point

		// iterate through all its neighbours
		for( int neibourIndex=0; neibourIndex<13; neibourIndex++ ) { 
			// the neighbour position
			int x, y, z; 
			Neighbour26::getNeigbour( neibourIndex, 
				dataPoints[site][0], dataPoints[site][1], dataPoints[site][2], 
				x, y, z ); 
			if( !indeces.isValid(x,y,z) ) continue; // not a valid position
			                                        // otherwise

			int site2 = indeces.at(x,y,z); 
			if( site2==-1 ) continue ; // not a neighbour
			                           // other wise, found a neighbour

			int l1 = labelings[site];
			int l2 = labelings[site2];

			if( l1==l2 ) continue; 

			double energy_smoothness_i = 0, energy_smoothness_j = 0; 
			compute_energy_smoothcost_for_pair( 
				lines[l1], lines[l2], 
				dataPoints[site], dataPoints[site2], 
				energy_smoothness_i, energy_smoothness_j ); 

			energy += energy_smoothness_i + energy_smoothness_j; 
		}
	}
	return energy; 
}

double compute_energy( 
	const vector<Vec3i>& dataPoints,
	const vector<int>& labelings, 
	const vector<Line3D*>& lines,
	const Data3D<int>& indeces )
{
	double datacost   = compute_energy_datacost( dataPoints, labelings, lines ); 
	double smoothcost = compute_energy_smoothcost( dataPoints, labelings, lines, indeces ); 
	return datacost + smoothcost; 
}

Mat compute_energy_matrix_datacost( 
	const vector<Vec3i>& dataPoints,
	const vector<int>& labelings, 
	const vector<Line3D*>& lines )
{
	Mat eng( (int) dataPoints.size(), 1, CV_64F ); 
	for( int site = 0; site < (int) dataPoints.size(); site++ ) {
		int label = labelings[site];
		eng.at<double>( site, 0 ) = sqrt( compute_energy_datacost_for_one( lines[label], dataPoints[site] ) ); 
	}
	return eng; 
}


void LevenburgMaquart::reestimate(const vector<Vec3i>& dataPoints,
	const vector<int>& labelings, 
	const vector<Line3D*>& lines,
	const Data3D<int>& indeces )
{
	double lambda = 1e-4; 

	const int numOfParametersPerLine = lines[0]->getNumOfParameters();
	const int numOfParametersTotal = (int) lines.size() * lines[0]->getNumOfParameters(); 

	for( int lmiter = 0; lambda < 10e50 && lambda > 10e-100 && lmiter<100000; lmiter++ ) { 
		cout << "Levenburg Maquart: " << lmiter << " Lambda: " << lambda << endl; 

		Mat energy_matrix = Mat( 0, 1, CV_64F );

		// Jacobian Matrix
		//  - # of cols: number of data points; 
		//  - # of rows: number of parameters for all the line models
		Mat Jacobian = Mat::zeros( 0, numOfParametersTotal, CV_64F ); 

		//// Construct Jacobian Matrix - for data cost
		energy_matrix = compute_energy_matrix_datacost( dataPoints, labelings, lines ); 
		Jacobian = Mat::zeros( (int) dataPoints.size(), numOfParametersTotal, CV_64F ); 
		for( int site=0; site < dataPoints.size(); site++ ) {
			int label = labelings[site]; 
			static const float delta = 0.01f; 

			double datacost_before = compute_energy_datacost_for_one( lines[label], dataPoints[site] ); 

			// compute the derivatives and construct Jacobian matrix
			for( int i=0; i < lines[label]->getNumOfParameters(); i++ ) {
				lines[label]->updateParameterWithDelta( i, delta ); 
				double datacost_new = compute_energy_datacost_for_one( lines[label], dataPoints[site] ); 
				Jacobian.at<double>( site, 6*label+i ) = 1.0 / delta * ( datacost_new - datacost_before ); 
				lines[label]->updateParameterWithDelta( i, -delta ); 
			}
		} // Contruct Jacobian matrix (2B Continue)


		//// Contruct Jacobian matrix (Continue) - for smooth cost
		//for( int site = 0; site < dataPoints.size(); site++ ) { // For each data point
		//	for( int neibourIndex=0; neibourIndex<13; neibourIndex++ ) { // find it's neighbour
		//		// the neighbour position
		//		int x, y, z; 
		//		Neighbour26::getNeigbour( neibourIndex, 
		//			dataPoints[site][0], dataPoints[site][1], dataPoints[site][2], 
		//			x, y, z ); 
		//		if( !indeces.isValid(x,y,z) ) continue; // not a valid position
		//		                                        // otherwise

		//		int site2 = indeces.at(x,y,z); 
		//		if( site2==-1 ) continue ; // not a neighbour
		//		                           // other wise, found a neighbour

		//		Mat JJ = Mat::zeros( 2, numOfParametersTotal, CV_64F ); 

		//		int l1 = labelings[site];
		//		int l2 = labelings[site2];
		//		
		//		if( l1==l2 ) continue; 

		//		double smoothcost_i_before = 0, smoothcost_j_before = 0;
		//		compute_energy_smoothcost_for_pair( 
		//			lines[l1], lines[l2], 
		//			dataPoints[site], dataPoints[site2], 
		//			smoothcost_i_before, smoothcost_j_before ); 
		//		
		//		// add more rows to energy_matrix according to smooth cost 
		//		energy_matrix.push_back( sqrt( smoothcost_i_before ) ); 
		//		energy_matrix.push_back( sqrt( smoothcost_j_before ) ); 

		//		// compute derivatives
		//		// Setting up J
		//		for( int label = 0; label < lines.size(); label++ ) { // for each label
		//			if( (l1==label) || (l2==label) ) {
		//				for( int i=0; i < numOfParametersPerLine; i++ ) {
		//					static const float delta = 0.1f; 

		//					// compute derivatives
		//					lines[label]->updateParameterWithDelta( i, delta ); 
		//					
		//					double smoothcost_i_new = 0, smoothcost_j_new = 0;
		//					compute_energy_smoothcost_for_pair( 
		//						lines[l1], lines[l2], 
		//						dataPoints[site], dataPoints[site2], 
		//						smoothcost_i_new, smoothcost_j_new ); 
		//					JJ.at<double>( 0, numOfParametersPerLine * label + i ) = 
		//						1.0 / delta * (smoothcost_i_new - smoothcost_i_before); 
		//					JJ.at<double>( 1, numOfParametersPerLine * label + i ) = 
		//						1.0 / delta * (smoothcost_j_new - smoothcost_j_before); 

		//					lines[label]->updateParameterWithDelta( i, -delta ); 
		//				}	
		//			}
		//		}
		//		// Add J1 and J2 to Jacobian matrix as an additional row
		//		Jacobian.push_back( JJ ); 
		//	} // for each pair of pi and pj
		//} // end of contruction of Jacobian Matrix



		Mat A = Jacobian.t() * Jacobian;
		A  = A + Mat::diag( lambda * Mat::ones(A.cols, 1, CV_64F) ); 

		Mat B = Jacobian.t() * energy_matrix; 

		Mat X; 
		cv::solve( A, -B, X, DECOMP_QR  ); 
		for( int i=0; i<X.rows; i++ ) {
			std::cout << std::setw(14) << std::scientific << X.at<double>(i) << "  ";
		}
		cout << endl;
		// Sleep(1000);
		


		double energy_before = compute_energy( dataPoints, labelings, lines, indeces );

		for( int label=0; label < lines.size(); label++ ) {
			for( int i=0; i < numOfParametersPerLine; i++ ) {
				const double& delta = X.at<double>( label * numOfParametersPerLine + i ); 
				lines[label]->updateParameterWithDelta( i, delta ); 
			}
		}

		double new_energy = compute_energy( dataPoints, labelings, lines, indeces );

		// the smaller lambda is, the faster it converges
		// the bigger lambda is, the slower it converges
		if( new_energy <= energy_before ) { // if energy is decreasing 
			cout << "-" << endl;
			energy_before = new_energy; 
			lambda *= 0.9; 
		} else {
			cout << "+" << endl;
			for( int label=0; label < lines.size(); label++ ) {
				for( int i=0; i < numOfParametersPerLine; i++ ) {
					const double& delta = X.at<double>( label * numOfParametersPerLine + i ); 
					lines[label]->updateParameterWithDelta( i, -delta ); 
				}
			}
			lambda *= 1.5; 
		}
	}
}
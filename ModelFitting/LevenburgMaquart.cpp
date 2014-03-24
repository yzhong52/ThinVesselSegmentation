#include "LevenburgMaquart.h"

#include <iostream> 
#include <iomanip>
#include <Windows.h>
using namespace std; 

#include "Line3D.h" 
#include "Neighbour26.h"
#include "Data3D.h" 

double compute_energy_datacost( 
	const vector<Vec3i>& dataPoints,
	const vector<int>& labelings, 
	const vector<Line3D*>& lines )
{
	double energy = 0; 

	for( int site = 0; site < (int) dataPoints.size(); site++ ) {
		int label = labelings[site];
		// log likelihood based on the distance
		double loglikelihood = lines[label]->loglikelihood( dataPoints[site] );
		energy += LOGLIKELIHOOD * loglikelihood; 
	}
	return energy; 
}


double compute_energy_smoothcost( 
	const vector<Vec3i>& dataPoints,
	const vector<int>& labelings, 
	const vector<Line3D*>& lines,
	const Data3D<int>& indeces )
{
	double energy = 0; 

	const int numOfParametersPerLine = lines[0]->getNumOfParameters();
	const int numOfParametersTotal = (int) lines.size() * lines[0]->getNumOfParameters(); 

	for( int site = 0; site < dataPoints.size(); site++ ) { // For each data point
		for( int nei=0; nei<13; nei++ ) { // find it's neighbour
			static int offx, offy, offz;
			Neighbour26::at( nei, offx, offy, offz ); 
			const int& x = dataPoints[site][0] + offx;
			const int& y = dataPoints[site][1] + offy;
			const int& z = dataPoints[site][2] + offz;
			if( !indeces.isValid(x,y,z) ) continue; 

			int site2 = indeces.at(x,y,z); 
			if( site2==-1 ) continue ; // not a neighbour
			// other wise, found a neighbour

			Mat JJ = Mat::zeros( 2, numOfParametersTotal, CV_64F ); 

			int l1 = labelings[site];
			int l2 = labelings[site2];

			// single projection
			Vec3f pi = lines[l1]->projection( dataPoints[site] ); 
			Vec3f pj = lines[l2]->projection( dataPoints[site2] ); 
			// double projection
			Vec3f pi1 = lines[l2]->projection( pi ); 
			Vec3f pj1 = lines[l1]->projection( pj ); 
			// distance vector
			Vec3f pipj  = pi - pj; 
			Vec3f pipi1 = pi - pi1; 
			Vec3f pjpj1 = pj - pj1;
			// distance
			float dist_pipj = pipj.dot(pipj); 
			float dist_pipi1 = pipi1.dot(pipi1); 
			float dist_pjpj1 = pjpj1.dot(pjpj1); 

			double energy_smoothness_i = dist_pipi1 / dist_pipj; 
			double energy_smoothness_j = dist_pjpj1 / dist_pipj; 
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
	double datacost = compute_energy_datacost( dataPoints, labelings, lines ); 
	double smoothcost = compute_energy_smoothcost( dataPoints, labelings, lines, indeces ); 
	return datacost + smoothcost; 
}

Mat computeenergy_matrix( 
	const vector<Vec3i>& dataPoints,
	const vector<int>& labelings, 
	const vector<Line3D*>& lines )
{
	Mat eng( (int) dataPoints.size(), 1, CV_64F ); 
	for( int site = 0; site < (int) dataPoints.size(); site++ ) {
		int label = labelings[site];
		// log likelihood based on the distance
		double loglikelihood = lines[label]->loglikelihood( dataPoints[site] );
		eng.at<double>( site, 0 ) = sqrt( LOGLIKELIHOOD * loglikelihood ); 
	}
	return eng; 
}


void LevenburgMaquart::reestimate(const vector<Vec3i>& dataPoints,
	const vector<int>& labelings, 
	const vector<Line3D*>& lines,
	const Data3D<int>& indeces )
{
	double lambda = 1e-2; 
	//double lambdaMultiplier = 1.0; 

	const int numOfParametersPerLine = lines[0]->getNumOfParameters();
	const int numOfParametersTotal = (int) lines.size() * lines[0]->getNumOfParameters(); 

	for( int lmiter = 0; lambda < 10e50 && lambda > 10e-100 && lmiter<100000; lmiter++ ) { 
		cout << "Levenburg Maquart: " << lmiter << " Lambda: " << lambda << endl; 

		Mat energy_matrix = Mat( 0, 1, CV_64F );

		// Jacobian Matrix
		//  - # of cols: number of data points; 
		//  - # of rows: number of parameters for all the line models
		Mat Jacobian = Mat::zeros( 0, numOfParametersTotal, CV_64F ); 

		if( LOGLIKELIHOOD > 1e-25 ) {
			//// Construct Jacobian Matrix - for data cost
			energy_matrix = computeenergy_matrix( dataPoints, labelings, lines ); 
			Jacobian = Mat::zeros( (int) dataPoints.size(), numOfParametersTotal, CV_64F ); 
			// Contruct Jacobian matrix
			for( int label=0; label < lines.size(); label++ ) {
				for( int site=0; site < dataPoints.size(); site++ ) {
					if( labelings[site] != label ) {
						for( int i=0; i < numOfParametersPerLine; i++ ) {
							Jacobian.at<double>( site, numOfParametersPerLine * label + i ) = 0; 
						}
					} 
					else 
					{
						static const float delta = 0.001f; 

						// TODO: move this out of the loop
						double energy_before_for_distance = compute_energy_datacost( dataPoints, labelings, lines ); 

						// compute the derivatives and construct Jacobian matrix
						for( int i=0; i < lines[label]->getNumOfParameters(); i++ ) {
							lines[label]->updateParameterWithDelta( i, delta ); 
							Jacobian.at<double>( site, 6*label+i ) = 1.0 / delta * ( compute_energy_datacost( dataPoints, labelings, lines ) - energy_before_for_distance ); 
							lines[label]->updateParameterWithDelta( i, -delta ); 
						}
					}
				}
			} // Contruct Jacobian matrix (2B Continue)
		}

		// Contruct Jacobian matrix (Continue) - for smooth cost
		for( int site = 0; site < dataPoints.size(); site++ ) { // For each data point
			for( int nei=0; nei<13; nei++ ) { // find it's neighbour
				static int offx, offy, offz;
				Neighbour26::at( nei, offx, offy, offz ); 
				const int& x = dataPoints[site][0] + offx;
				const int& y = dataPoints[site][1] + offy;
				const int& z = dataPoints[site][2] + offz;
				if( !indeces.isValid(x,y,z) ) continue; 
				
				int site2 = indeces.at(x,y,z); 
				if( site2==-1 ) continue ; // not a neighbour
				// other wise, found a neighbour

				Mat JJ = Mat::zeros( 2, numOfParametersTotal, CV_64F ); 

				int l1 = labelings[site];
				int l2 = labelings[site2];
				
				// single projection
				Vec3f pi = lines[l1]->projection( dataPoints[site] ); 
				Vec3f pj = lines[l2]->projection( dataPoints[site2] ); 
				// double projection
				Vec3f pi1 = lines[l2]->projection( pi ); 
				Vec3f pj1 = lines[l1]->projection( pj ); 
				// distance vector
				Vec3f pipj  = pi - pj; 
				Vec3f pipi1 = pi - pi1; 
				Vec3f pjpj1 = pj - pj1;
				// distance
				float dist_pipj = pipj.dot(pipj); 
				float dist_pipi1 = pipi1.dot(pipi1); 
				float dist_pjpj1 = pjpj1.dot(pjpj1); 

				double energy_smoothness_i = dist_pipi1 / dist_pipj; 
				double energy_smoothness_j = dist_pjpj1 / dist_pipj; 

				// add more rows to energy_matrix according to smooth cost 
				energy_matrix.push_back( sqrt( energy_smoothness_i ) ); 
				energy_matrix.push_back( sqrt( energy_smoothness_j ) ); 

				// compute derivatives
				// Setting up J
				for( int label = 0; label < lines.size(); label++ ) { // for each label
					if( (l1==label) || (l2==label) ) {
						for( int i=0; i < numOfParametersPerLine; i++ ) {
							static const float delta = 0.001f; 

							lines[label]->updateParameterWithDelta( i, delta ); 
							
							// single projection
							Vec3f pi = lines[l1]->projection( dataPoints[site] ); 
							Vec3f pj = lines[l2]->projection( dataPoints[site2] ); 
							// double projection
							Vec3f pi1 = lines[l2]->projection( pi ); 
							Vec3f pj1 = lines[l1]->projection( pj ); 
							// distance vector
							Vec3f pipj  = pi - pj; 
							Vec3f pipi1 = pi - pi1; 
							Vec3f pjpj1 = pj - pj1;
							// distance
							float dist_pipj = pipj.dot(pipj); 
							float dist_pipi1 = pipi1.dot(pipi1); 
							float dist_pjpj1 = pjpj1.dot(pjpj1); 

							// compute derivatives
							JJ.at<double>( 0, numOfParametersPerLine * label + i ) = 
								dist_pipi1 / dist_pipj - energy_smoothness_i; 
							JJ.at<double>( 1, numOfParametersPerLine * label + i ) = 
								dist_pjpj1 / dist_pipj - energy_smoothness_j; 

							lines[label]->updateParameterWithDelta( i, -delta ); 
						}
						
					} else {
						for( int i=0; i < numOfParametersPerLine; i++ ) {
							JJ.at<double>( 0, numOfParametersPerLine * label + i ) = 0; 
							JJ.at<double>( 1, numOfParametersPerLine * label + i ) = 0; 
						}
					}
					
				}
				// Add J1 and J2 to Jacobian matrix as an additional row
				Jacobian.push_back( JJ ); 
			} // for each pair of pi and pj
		} // end of contruction of Jacobian Matrix


		Mat A = Jacobian.t() * Jacobian;
		A  = A + Mat::diag( lambda * Mat::ones(A.cols, 1, CV_64F) ); 

		Mat B = Jacobian.t() * energy_matrix; 

		Mat X; 
		cv::solve( A, -B, X, DECOMP_QR  ); 
		//for( int i=0; i<X.rows; i++ ) {
		//	std::cout << std::setw(14) << std::scientific << X.at<double>(i) << "  ";
		//}
		//cout << endl;
		//
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
			lambda *= 0.99; 
		} else {
			cout << "+" << endl;
			for( int label=0; label < lines.size(); label++ ) {
				for( int i=0; i < numOfParametersPerLine; i++ ) {
					const double& delta = X.at<double>( label * numOfParametersPerLine + i ); 
					lines[label]->updateParameterWithDelta( i, -delta ); 
				}
			}
			lambda *= 1.01; 
		}
		
		// Sleep(10);  // TODO: this is only for debuging 
	}
}
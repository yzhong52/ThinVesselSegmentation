#include "LevenburgMaquart.h"

#include <iostream> 
#include <iomanip>
#include <Windows.h>
using namespace std; 

#include "Line3D.h" 

double computeEnergy( 
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


Mat computeEnergyMatrix( 
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
	const vector<Line3D*>& lines ){
	double lambda = 1e4; 

	// TODO: this line is not necessary if we have run graph cut
	double energy_before = computeEnergy( dataPoints, labelings, lines ); 

	for( int lmiter = 0; lambda < 10e10; lmiter++ ) { 
		cout << "Levenburg Maquart: " << lmiter << " Lambda: " << lambda << endl; 

		// there are six parameters
		// Jacobian Matrix ( # of cols: number of data points; # of rows: number of parameters of each line models)? 
		Mat Jacobian = Mat::zeros(
			(int) dataPoints.size(), 
			(int) lines.size() * lines[0]->getNumOfParameters(),
			CV_64F ); 

		// Contruct Jacobian matrix
		for( int label=0; label < lines.size(); label++ ) {
			for( int site=0; site < dataPoints.size(); site++ ) {
				if( labelings[site] != label ) {
					Jacobian.at<double>( site, 6*label ) = 0; 
					Jacobian.at<double>( site, 6*label+1 ) = 0; 
					Jacobian.at<double>( site, 6*label+2 ) = 0; 
					Jacobian.at<double>( site, 6*label+3 ) = 0; 
					Jacobian.at<double>( site, 6*label+4 ) = 0; 
					Jacobian.at<double>( site, 6*label+5 ) = 0; 
				} 
				else 
				{
					static const float delta = 0.001f; 

					// compute the derivatives and construct Jacobian matrix
					for( int i=0; i < lines[label]->getNumOfParameters(); i++ ) {
						lines[label]->updateParameterWithDelta( i, delta ); 
						Jacobian.at<double>( site, 6*label+i ) = 
							1.0 / delta * ( computeEnergy( dataPoints, labelings, lines ) - energy_before ); 
						lines[label]->updateParameterWithDelta( i, -delta ); 
					}
				}
			}
		} // end of contruction of Jacobian Matrix

		Mat A = Jacobian.t() * Jacobian; 

		A = A + Mat::diag( lambda * Mat::ones(A.cols, 1, CV_64F) ); 


		Mat B = Jacobian.t() * computeEnergyMatrix( dataPoints, labelings, lines ); 

		Mat X; 
		cv::solve( A, -B, X, DECOMP_QR  ); 
		for( int i=0; i<X.rows; i++ ) {
			std::cout << std::setw(14) << std::scientific << X.at<double>(i) << "  ";
		}
		cout << endl;

		for( int label=0; label < lines.size(); label++ ) {
			for( int i=0; i < lines[label]->getNumOfParameters(); i++ ) {
				const double& delta = X.at<double>( label * (int) lines.size() + i ); 
				lines[label]->updateParameterWithDelta( i, delta ); 
			}
		}
		double new_energy = computeEnergy( dataPoints, labelings, lines );
		if( new_energy < energy_before ) { // if energy is decreasing 
			// the smaller lambda is, the faster it converges
			lambda /= 2; 
			energy_before = new_energy; 
		} else {
			for( int label=0; label < lines.size(); label++ ) {
				for( int i=0; i < lines[label]->getNumOfParameters(); i++ ) {
					const double& delta = X.at<double>( label * (int) lines.size() + i ); 
					lines[label]->updateParameterWithDelta( i, -delta ); 
				}
			}
			// the bigger lambda is, the slower it converges
			lambda *= 2; 
		}

		// Sleep(300);  // TODO: this is only for debuging 
	}
}
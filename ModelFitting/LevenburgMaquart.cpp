#include "LevenburgMaquart.h"

#include <iostream> 
#include <iomanip>
#include <Windows.h>
#include <limits>
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
	double& smooth_cost_i, 
	double& smooth_cost_j )
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
	double dist_pipj  = pipj.dot(pipj); 
	double dist_pipi1 = pipi1.dot(pipi1); 
	double dist_pjpj1 = pjpj1.dot(pjpj1); 

	if( dist_pipj < 1e-10 ) dist_pipj = 1e-10; 
	
	if( dist_pipi1 < 1e-10 ) smooth_cost_i = 0; 
	else smooth_cost_i = PAIRWISESMOOTH * dist_pipi1 / dist_pipj; 
	
	if( dist_pjpj1 < 1e-10 ) smooth_cost_j = 0; 
	else smooth_cost_j = PAIRWISESMOOTH * dist_pjpj1 / dist_pipj; 
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






void  projection_jacobians( 
	const Vec3f X1, const Vec3f X2,                  // two end points of a line
	const Mat nablaX1, const Mat nablaX2,          // Jacobians of the end points of the line
	const Vec3i& tildeP, const Mat nablaTildeP,   // a point, and the Jacobian of the point
	Vec3f& P, Mat& nablaP )
{	
	// Convert the end points into matrix form
	const Mat MX1 = (Mat_<double>(3,1) << double(X1[0]), double(X1[1]), double(X1[2]) ); 
	const Mat MX2 = (Mat_<double>(3,1) << double(X2[0]), double(X2[1]), double(X2[2]) ); 
	// convert the observe points into matrix form
	const Mat MTildeP = (Mat_<double>(3,1) << double(tildeP[0]), double(tildeP[1]), double(tildeP[2]) ); 

	// Assume that: P = T * X1 + (1-T) * X2
	const double A = ( Vec3f(tildeP) - X2 ).dot( X1 - X2 );
	const double B = ( X1 - X2 ).dot( X1 - X2 );
	const double T = A / B;
	
	// Compute the Jacobian matrix for Ai, Bi and Aj, Bj
	const Mat nablaA = ( nablaTildeP - nablaX2 ) * ( MX1 - MX2 ) + (nablaX1 - nablaX2) * ( MTildeP - MX2 );
	const Mat nablaB = 2 * ( nablaX1 - nablaX2 ) * ( MX1 - MX2 ); 
	
	// Compute the Jacobian matrix for Ti and Tj
	// N * 1 matrices
	const Mat nablaT = ( nablaA * B - nablaB * A ) / ( B * B );

	// convert the projection points into matrix form
	const Mat MP = (Mat_<double>(3,1) << double(P[0]), double(P[1]), double(P[2]) ); 
	
	// Compute the projection point
	P = T * X1 + (1-T) * X2; 

	// And the Jacobian matrix of the projection point
	// N * 3  matrices
	nablaP = nablaT * MX1.t() + nablaX1 * T + nablaX2 * (1-T) - nablaT * MX2.t();
}


Mat compute_energy_datacost_derivative_for_one( const Line3D* l,  const Vec3i tildeP ) {
	Vec3f X1, X2; 
	l->getEndPoints( X1, X2 ); 
	static const Mat nablaX1 = (Mat_<double>(6, 3) << 
		1.0, 0.0, 0.0,
		0.0, 1.0, 0.0,
		0.0, 0.0, 1.0,
		0.0, 0.0, 0.0,
		0.0, 0.0, 0.0,
		0.0, 0.0, 0.0 );
	static const Mat nablaX2 = (Mat_<double>(6, 3) << 
		0.0, 0.0, 0.0,
		0.0, 0.0, 0.0,
		0.0, 0.0, 0.0,
		1.0, 0.0, 0.0,
		0.0, 1.0, 0.0,
		0.0, 0.0, 1.0 );

	//Mat nablaA( 6, 1, CV_64F );
	//nablaA.at<double>(0) =  tildeP[0] - X2[0]; 
	//nablaA.at<double>(1) =  tildeP[1] - X2[1]; 
	//nablaA.at<double>(2) =  tildeP[2] - X2[2]; 
	//nablaA.at<double>(3) = -tildeP[0] - X1[0] + 2 * X2[0]; 
	//nablaA.at<double>(4) = -tildeP[1] - X1[1] + 2 * X2[1]; 
	//nablaA.at<double>(5) = -tildeP[2] - X1[2] + 2 * X2[2]; 

	//Mat nablaB( 6, 1, CV_64F ); 
	//nablaB.at<double>(0) = 2 * ( X1[0] - X2[0] ); 
	//nablaB.at<double>(1) = 2 * ( X1[1] - X2[1] ); 
	//nablaB.at<double>(2) = 2 * ( X1[2] - X2[2] ); 
	//nablaB.at<double>(3) = 2 * ( X2[0] - X1[0] ); 
	//nablaB.at<double>(4) = 2 * ( X2[1] - X1[1] ); 
	//nablaB.at<double>(5) = 2 * ( X2[2] - X1[2] ); 

	//double A = ( Vec3f(tildeP) - X2 ).dot( X1 - X2 );
	//double B = ( X1 - X2 ).dot( X1 - X2 );

	//Mat nablaT = (nablaA * B - nablaB * A ) / (B * B); // 6 * 1 matrix
	//
	//double T = A / B; // B is the length of a line (won't be zero) 

	//Mat MX1 = Mat::zeros( 3,1, CV_64F ); 
	//Mat MX2 = Mat::zeros( 3,1, CV_64F ); 
	//for( int i=0; i<3; i++ ) {
	//	MX1.at<double>(i)   = X1[i];
	//	MX2.at<double>(i) = X2[i];
	//}

	//// 6 * 3 
	//Mat nablaP = nablaT * MX1.t() + nablaX1 * T + nablaX2 * (1-T) - nablaT * MX2.t();
	//
	//Vec3d tildeP_P = Vec3d(tildeP) - Vec3d( T * X1 + (1-T) * X2 ); 
	//double tildaP_P_lenght = max( 1e-10, sqrt( tildeP_P.dot(tildeP_P) ) ); // l->distanceToLine( tildeP );

	////cout << nablaP << endl;

	// Mat M_TildeP_P = (Mat_<double>(3, 1) << tildeP_P[0], tildeP_P[1], tildeP_P[2]); 

	
	Vec3f P; 
	Mat nablaP; 
	projection_jacobians( X1, X2, nablaX1, nablaX2, tildeP, Mat::zeros( 6, 3, CV_64F), P, nablaP );
	Vec3d tildeP_P = Vec3d(tildeP) - Vec3d(P); 
	double tildaP_P_lenght = max( 1e-10, sqrt( tildeP_P.dot(tildeP_P) ) ); // l->distanceToLine( tildeP );
	Mat M_TildeP_P = (Mat_<double>(3, 1) << tildeP_P[0], tildeP_P[1], tildeP_P[2]); 
	
	return -nablaP * M_TildeP_P / tildaP_P_lenght * LOGLIKELIHOOD; 
}




Mat compute_energy_smoothcost_derivative_for_pair( const Line3D* li,  const Line3D* lj, const Vec3i& tildePi, const Vec3i& tildePj ) {
	// Get the end points of the line 
	// [Xi1, Xi2]
	// [Xj1, Xj2]
	Vec3f Xi1, Xi2, Xj1, Xj2; 
	li->getEndPoints( Xi1, Xi2 ); 
	lj->getEndPoints( Xj1, Xj2 ); 

	// Convert the end points into matrix form
	Mat MXi1 = Mat::zeros( 3,1, CV_64F ); 
	Mat MXi2 = Mat::zeros( 3,1, CV_64F ); 
	Mat MXj1 = Mat::zeros( 3,1, CV_64F ); 
	Mat MXj2 = Mat::zeros( 3,1, CV_64F ); 
	for( int i=0; i<3; i++ ) {
		MXi1.at<double>(i) = Xi1[i];
		MXi2.at<double>(i) = Xi2[i];
		MXj1.at<double>(i) = Xj1[i];
		MXj2.at<double>(i) = Xj2[i];
	}

	// Compute the derivatives of the end points over 
	// the 12 parameters
	static Mat nablaXi1 = Mat::zeros(12, 3, CV_64F);
	static Mat nablaXi2 = Mat::zeros(12, 3, CV_64F);
	static Mat nablaXj1 = Mat::zeros(12, 3, CV_64F);
	static Mat nablaXj2 = Mat::zeros(12, 3, CV_64F);
	for( int i=0; i<3; i++ ) {
		nablaXi1.at<double>(i+0,i) = 1; 
		nablaXi2.at<double>(i+3,i) = 1; 
		nablaXj1.at<double>(i+6,i) = 1; 
		nablaXj2.at<double>(i+9,i) = 1; 
	}
	//cout << nablaXi1 << endl << endl; 
	//cout << nablaXi2 << endl << endl; 
	//cout << nablaXj1 << endl << endl; 
	//cout << nablaXj2 << endl << endl; 

	// convert the observe points into matrix form
	Mat MTildePi = (Mat_<double>(3,1) << 
		double(tildePi[0]), 
		double(tildePi[1]), 
		double(tildePi[2]) ); 
	Mat MTildePj = (Mat_<double>(3,1) << 
		double(tildePj[0]), 
		double(tildePj[1]), 
		double(tildePj[2]) ); 
	//cout << MTildePi << endl << endl; 
	//cout << MTildePj << endl << endl; 

	// Assume that:
	//     Pi = Ti * Xi1 + (1-Ti) * Xi2
	//     Pj = Tj * Xj1 + (1-Tj) * Xj2
	
	const double Ai = ( Vec3f(tildePi) - Xi2 ).dot( Xi1 - Xi2 );
	const double Bi = ( Xi1 - Xi2 ).dot( Xi1 - Xi2 );
	const double Ti = Ai / Bi;
	
	const double Aj = ( Vec3f(tildePj) - Xj2 ).dot( Xj1 - Xj2 );
	const double Bj = ( Xj1 - Xj2 ).dot( Xj1 - Xj2 );
	const double Tj = Aj / Bj;

	// Compute the Jacobian matrix for Ai, Bi and Aj, Bj
	const Mat nablaAi = -nablaXi2 * ( MXi1 - MXi2 ) + (nablaXi1 - nablaXi2) * ( MTildePi - MXi2 );
	const Mat nablaBi = 2 * ( nablaXi1 - nablaXi2 ) * ( MXi1 - MXi2 ); 
	const Mat nablaAj = -nablaXj2 * ( MXj1 - MXj2 ) + (nablaXj1 - nablaXj2) * ( MTildePj - MXj2 ); 
	const Mat nablaBj = 2 * ( nablaXj1 - nablaXj2 ) * ( MXj1 - MXj2 ); 

	// Compute the Jacobian matrix for Ti and Tj
	// 12 * 1 matrices
	const Mat nablaTi = ( nablaAi * Bi - nablaBi * Ai ) / ( Bi * Bi );
	const Mat nablaTj = ( nablaAj * Bj - nablaBj * Aj ) / ( Bj * Bj );

	// Compute the projection point
	const Vec3f Pi = Ti * Xi1 + (1-Ti) * Xi2; 
	const Vec3f Pj = Tj * Xj1 + (1-Tj) * Xj2; 

	//cout << endl << "nablaTi = "; 
	//for( int i=0; i<nablaTi.rows; i++ ) {	
	//	if( i%3==0 ) cout << endl; 
	//	std::cout << std::setw(14) << std::scientific << nablaTi.at<double>(i ) << "  ";
	//}
	//cout << endl;

	//cout << endl << "nablaTj = "; 
	//for( int i=0; i<nablaTj.rows; i++ ) {	
	//	if( i%3==0 ) cout << endl; 
	//	std::cout << std::setw(14) << std::scientific << nablaTj.at<double>(i ) << "  ";
	//}
	//cout << endl;

	// convert the projection points into matrix form
	const Mat MPi = (Mat_<double>(3,1) << 
		double(Pi[0]), 
		double(Pi[1]), 
		double(Pi[2]) ); 
	const Mat MPj = (Mat_<double>(3,1) << 
		double(Pj[0]), 
		double(Pj[1]), 
		double(Pj[2]) ); 

	// And the Jacobian matrix of the projection point
	// 12 * 3  matrices
	const Mat nablaPi 
		= nablaTi * MXi1.t() 
		+ nablaXi1 * Ti 
		+ nablaXi2 * (1-Ti) 
		- nablaTi * MXi2.t();
	const Mat nablaPj 
		= nablaTj * MXj1.t() 
		+ nablaXj1 * Tj 
		+ nablaXj2 * (1-Tj) 
		- nablaTj * MXj2.t();
	
	//cout << endl << "nablaPi = "; 
	//for( int i=0; i<nablaPi.rows; i++ ) for( int j=0; j<nablaPi.cols; j++ ) {	
	//	if( j%3==0 ) cout << endl; 
	//	std::cout << std::setw(14) << std::scientific << nablaPi.at<double>(i, j) << "  ";
	//}
	//cout << endl;

	//cout << endl << "nablaPj = "; 
	//for( int i=0; i<nablaPj.rows; i++ ) for( int j=0; j<nablaPj.cols; j++ ) {	
	//	if( j%3==0 ) cout << endl; 
	//	std::cout << std::setw(14) << std::scientific << nablaPj.at<double>(i, j) << "  ";
	//}
	//cout << endl;



	const double  Ai_prime = ( Vec3f(Pi) - Xj2 ).dot( Xj1 - Xj2 );
	const double& Bi_prime = Bj;
	const double  Ti_prime = Ai_prime / Bi_prime; 

	const double  Aj_prime = ( Vec3f(Pj) - Xi2 ).dot( Xi1 - Xi2 );
	const double& Bj_prime = Bi;
	const double  Tj_prime = Aj_prime / Bj_prime; 

	// Compute the projeciton point again (onto each other's line)
	const Vec3f Pi_prime = Ti_prime * Xj1 + ( 1-Ti_prime ) * Xj2; 
	const Vec3f Pj_prime = Tj_prime * Xi1 + ( 1-Tj_prime ) * Xi2; 

	// convert the projection points into matrix form
	const Mat MPi_prime = (Mat_<double>(3,1) << 
		double(Pi_prime[0]), 
		double(Pi_prime[1]), 
		double(Pi_prime[2]) ); 
	const Mat MPj_prime = (Mat_<double>(3,1) << 
		double(Pj_prime[0]), 
		double(Pj_prime[1]), 
		double(Pj_prime[2]) ); 

	const Mat nablaAi_prime = nablaPi * ( MXj1 - MXj2 ) + nablaAj; 
	const Mat nablaBi_prime = nablaBj; 
	const Mat nablaAj_prime = nablaPj * ( MXi1 - MXi2 ) + nablaAi; 
	const Mat nablaBj_prime = nablaBi;

	const Mat nablaTi_prime = ( nablaAi_prime * Bi_prime - nablaBi_prime * Ai_prime ) / ( Bi_prime * Bi_prime );
	const Mat nablaTj_prime = ( nablaAj_prime * Bj_prime - nablaBj_prime * Aj_prime ) / ( Bj_prime * Bj_prime ); 

	// 12 * 3 matrices 
	Mat nabla_Pi_prime 
		= nablaTi_prime * MXj1.t() 
		+ nablaXj1 * Ti_prime 
		+ nablaXj2 * ( 1 - Ti_prime ) 
		- nablaTi_prime * MXj2.t();
	Mat nabla_Pj_prime 
		= nablaTj_prime * MXi1.t() 
		+ nablaXi1 * Tj_prime
		+ nablaXi2 * ( 1 - Tj_prime ) 
		- nablaTj_prime * MXi2.t();

	const double dist_pi_pj2       = max(1e-27, (Pi-Pj).dot(Pi-Pj) ); 
	const double dist_pi_pi_prime2 = max(1e-27, (Pi-Pi_prime).dot(Pi-Pi_prime) ); 
	const double dist_pj_pj_prime2 = max(1e-27, (Pj-Pj_prime).dot(Pj-Pj_prime) ); 
	const double dist_pi_pj        = sqrt( dist_pi_pj2 );
	const double dist_pi_pi_prime  = sqrt( dist_pi_pi_prime2 ); 
	const double dist_pj_pj_prime  = sqrt( dist_pj_pj_prime2 ); 

	Mat nabla_pi_pi_prime = (nablaPi - nabla_Pi_prime) * ( MPi - MPi_prime ) / dist_pi_pi_prime; 
	Mat nabla_pj_pj_prime = (nablaPj - nabla_Pj_prime) * ( MPj - MPj_prime ) / dist_pj_pj_prime; 
	Mat nabla_pi_pj       = (nablaPi - nablaPj) * ( MPi - MPj ) / dist_pi_pj; 

	Mat res( 0, 12, CV_64F ); 
	Mat row1 = nabla_pi_pi_prime * dist_pi_pj - nabla_pi_pj * dist_pi_pi_prime; 
	Mat row2 = nabla_pj_pj_prime * dist_pi_pj - nabla_pi_pj * dist_pj_pj_prime; 


	//for( int i=0; i<row1.rows; i++ ) {	
	//	if( i%3==0 ) cout << endl; 
	//	std::cout << std::setw(14) << std::scientific << row1.at<double>(i ) << "  ";
	//}
	//cout << endl;
	// 
	//for( int i=0; i<row2.rows; i++ ) {	
	//	if( i%3==0 ) cout << endl; 
	//	std::cout << std::setw(14) << std::scientific << row2.at<double>(i ) << "  ";
	//}
	//cout << endl; 

	//cout << endl << "nabla_pi_pi_prime = "; 
	//for( int i=0; i<nabla_pi_pi_prime.rows; i++ ) {	
	//	if( i%3==0 ) cout << endl; 
	//	std::cout << std::setw(14) << std::scientific << nabla_pi_pi_prime.at<double>(i ) << "  ";
	//}
	//cout << endl;

	//cout << endl << "nabla_pj_pj_prime = "; 
	//for( int i=0; i<nabla_pj_pj_prime.rows; i++ ) {	
	//	if( i%3==0 ) cout << endl; 
	//	std::cout << std::setw(14) << std::scientific << nabla_pj_pj_prime.at<double>(i ) << "  ";
	//}
	//cout << endl;

	//cout << endl << "nabla_pi_pj = "; 
	//for( int i=0; i<nabla_pi_pj.rows; i++ ) {	
	//	if( i%3==0 ) cout << endl; 
	//	std::cout << std::setw(14) << std::scientific << nabla_pi_pj.at<double>(i ) << "  ";
	//}





	res.push_back( Mat( row1.t() ) );
	res.push_back( Mat( row2.t() ) );
	res = res / dist_pi_pj2; 
	
	//for( int i=0; i<res.cols; i++ ) {
	//	std::cout << std::setw(14) << std::scientific << res.at<double>(0, i) << "  ";
	//	if( (i+1)%3 == 0 ) { cout << endl; }
	//}

	return res; 
}




Mat compute_energy_smoothcost_derivative_for_pair2( const Line3D* li,  const Line3D* lj, const Vec3i& tildePi, const Vec3i& tildePj ) {
	Vec3f Xi1, Xi2, Xj1, Xj2; 
	li->getEndPoints( Xi1, Xi2 ); 
	lj->getEndPoints( Xj1, Xj2 ); 

	// Compute the derivatives of the end points over 
	// the 12 parameters
	static Mat nablaXi1 = Mat::zeros(12, 3, CV_64F);
	static Mat nablaXi2 = Mat::zeros(12, 3, CV_64F);
	static Mat nablaXj1 = Mat::zeros(12, 3, CV_64F);
	static Mat nablaXj2 = Mat::zeros(12, 3, CV_64F);
	for( int i=0; i<3; i++ ) {
		nablaXi1.at<double>(i+0,i) = 1; 
		nablaXi2.at<double>(i+3,i) = 1; 
		nablaXj1.at<double>(i+6,i) = 1; 
		nablaXj2.at<double>(i+9,i) = 1; 
	}

	Vec3f Pi, Pj, Pi_prime, Pj_prime;
	Mat nablaPi, nablaPj, nablaPi_prime, nablaPj_prime; 

	projection_jacobians( Xi1, Xi2, nablaXi1, nablaXi2, tildePi, Mat::zeros( 12, 3, CV_64F), Pi, nablaPi );
	projection_jacobians( Xj1, Xj2, nablaXj1, nablaXj2, tildePj, Mat::zeros( 12, 3, CV_64F), Pj, nablaPj );
	projection_jacobians( Xj1, Xj2, nablaXj1, nablaXj2, Pi, nablaPi, Pi_prime, nablaPi_prime );
	projection_jacobians( Xi1, Xi2, nablaXi1, nablaXi2, Pj, nablaPj, Pj_prime, nablaPj_prime );

	const double dist_pi_pj2       = max(1e-27, (Pi-Pj).dot(Pi-Pj) ); 
	const double dist_pi_pi_prime2 = max(1e-27, (Pi-Pi_prime).dot(Pi-Pi_prime) ); 
	const double dist_pj_pj_prime2 = max(1e-27, (Pj-Pj_prime).dot(Pj-Pj_prime) ); 
	const double dist_pi_pj        = sqrt( dist_pi_pj2 );
	const double dist_pi_pi_prime  = sqrt( dist_pi_pi_prime2 ); 
	const double dist_pj_pj_prime  = sqrt( dist_pj_pj_prime2 ); 

	// Convert the end points into matrix form
	const Mat MPi = (Mat_<double>(3,1) << double(Pi[0]), double(Pi[1]), double(Pi[2]) ); 
	const Mat MPj = (Mat_<double>(3,1) << double(Pj[0]), double(Pj[1]), double(Pj[2]) ); 
	const Mat MPi_prime = (Mat_<double>(3,1) << double(Pi_prime[0]), double(Pi_prime[1]), double(Pi_prime[2]) ); 
	const Mat MPj_prime = (Mat_<double>(3,1) << double(Pj_prime[0]), double(Pj_prime[1]), double(Pj_prime[2]) ); 
	
	const Mat nabla_pi_pi_prime = (nablaPi - nablaPi_prime) * ( MPi - MPi_prime ) / dist_pi_pi_prime; 
	const Mat nabla_pj_pj_prime = (nablaPj - nablaPj_prime) * ( MPj - MPj_prime ) / dist_pj_pj_prime; 
	const Mat nabla_pi_pj       = (nablaPi - nablaPj) * ( MPi - MPj ) / dist_pi_pj; 

	Mat res( 0, 12, CV_64F ); 
	const Mat row1 = nabla_pi_pi_prime * dist_pi_pj - nabla_pi_pj * dist_pi_pi_prime; 
	const Mat row2 = nabla_pj_pj_prime * dist_pi_pj - nabla_pi_pj * dist_pj_pj_prime; 

	res.push_back( Mat( row1.t() ) );
	res.push_back( Mat( row2.t() ) );
	res = res / dist_pi_pj2 * PAIRWISESMOOTH; 
	
	//for( int i=0; i<res.cols; i++ ) {
	//	std::cout << std::setw(14) << std::scientific << res.at<double>(0, i) << "  ";
	//	if( (i+1)%3 == 0 ) { cout << endl; }
	//}

	return res; 
}






void LevenburgMaquart::reestimate(const vector<Vec3i>& dataPoints,
	const vector<int>& labelings, 
	const vector<Line3D*>& lines,
	const Data3D<int>& indeces )
{
	double lambda = 1e2; 

	const int numOfParametersPerLine = lines[0]->getNumOfParameters();
	const int numOfParametersTotal = (int) lines.size() * lines[0]->getNumOfParameters(); 

	for( int lmiter = 0; lambda < 10e50 && lambda > 10e-100 && lmiter<100000; lmiter++ ) { 
		// cout << "Levenburg Maquart: " << lmiter << " Lambda: " << lambda << endl; 

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
			static const float delta = 0.0003f; 
			// compute the derivatives and construct Jacobian matrix
		 double datacost_before = compute_energy_datacost_for_one( lines[label], dataPoints[site] ); 
			// Computing Derivative Nemerically
			/*for( int i=0; i < numOfParametersPerLine; i++ ) {
				lines[label]->updateParameterWithDelta( i, delta ); 
				double datacost_new = compute_energy_datacost_for_one( lines[label], dataPoints[site] ); 
				Jacobian.at<double>( site, 6*label+i ) = 1.0 / delta * ( datacost_new - datacost_before ); 
				lines[label]->updateParameterWithDelta( i, -delta ); 
			}
			*/
			// Computing Derivative Analytically
			Mat m = compute_energy_datacost_derivative_for_one( lines[label], dataPoints[site] ); 
			for( int i=0; i < numOfParametersPerLine; i++ ) {
				Jacobian.at<double>( site, 6*label+i ) = m.at<double>(i); 
			}
		} // Contruct Jacobian matrix (2B Continue)


		// Contruct Jacobian matrix (Continue) - for smooth cost
		for( int site = 0; site < dataPoints.size(); site++ ) { // For each data point
			for( int neibourIndex=0; neibourIndex<13; neibourIndex++ ) { // find it's neighbour
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

				Mat JJ = Mat::zeros( 2, numOfParametersTotal, CV_64F ); 

				int l1 = labelings[site];
				int l2 = labelings[site2];
				
				if( l1==l2 ) continue; // TODO

				double smoothcost_i_before = 0, smoothcost_j_before = 0;
				compute_energy_smoothcost_for_pair( 
					lines[l1], lines[l2], 
					dataPoints[site], dataPoints[site2], 
					smoothcost_i_before, smoothcost_j_before ); 
				
				// add more rows to energy_matrix according to smooth cost 
				energy_matrix.push_back( sqrt( smoothcost_i_before ) ); 
				energy_matrix.push_back( sqrt( smoothcost_j_before ) ); 

				// compute derivatives numerically
				// Setting up J
				// Computing derivative of pair-wise smooth cost numerically
				for( int label = 0; label < lines.size(); label++ ) { // for each label
					if( (l1==label) || (l2==label) ) {
						for( int i=0; i < numOfParametersPerLine; i++ ) {
							static const float delta = 0.0001f; 
							// compute derivatives
							lines[label]->updateParameterWithDelta( i, delta ); 
							double smoothcost_i_new = 0, smoothcost_j_new = 0;
							compute_energy_smoothcost_for_pair( 
								lines[l1], lines[l2], 
								dataPoints[site], dataPoints[site2], 
								smoothcost_i_new, smoothcost_j_new ); 
							JJ.at<double>( 0, numOfParametersPerLine * label + i ) = 
								1.0 / delta * (smoothcost_i_new - smoothcost_i_before); 
							JJ.at<double>( 1, numOfParametersPerLine * label + i ) = 
								1.0 / delta * (smoothcost_j_new - smoothcost_j_before); 
							lines[label]->updateParameterWithDelta( i, -delta ); 
						}	
					}
				}
				


				//cout << endl << "JJ = " << endl;
				//for( int i=0; i<JJ.rows; i++ ) {
				//	for( int j=0; j<JJ.cols; j++ ) {	
				//		if( j%3==0 ) cout << endl; 
				//		std::cout << std::setw(14) << std::scientific << JJ.at<double>(i, j) << "  ";
				//	}
				//	cout << endl; 
				//}
				//cout << endl;

				//// Computing derivative of pair-wise smooth cost analytically
				Mat tempJJ = compute_energy_smoothcost_derivative_for_pair2( 
					lines[l1], lines[l2], 
					dataPoints[site], dataPoints[site2] ); 
				
				for( int i=0; i < numOfParametersPerLine; i++ ) { 
					JJ.at<double>( 0, numOfParametersPerLine * l1 + i ) = tempJJ.at<double>(0, i); 
					JJ.at<double>( 0, numOfParametersPerLine * l2 + i ) = tempJJ.at<double>(0, i + numOfParametersPerLine); 
					JJ.at<double>( 1, numOfParametersPerLine * l1 + i ) = tempJJ.at<double>(1, i); 
					JJ.at<double>( 1, numOfParametersPerLine * l2 + i ) = tempJJ.at<double>(1, i + numOfParametersPerLine ); 
				}
				

				//Mat tempJJ2 = compute_energy_smoothcost_derivative_for_pair2( 
				//	lines[l1], lines[l2], 
				//	dataPoints[site], dataPoints[site2] ); 

				//cout << endl << "tempJJ = " << endl;
				//for( int i=0; i<tempJJ.rows; i++ ) {
				//	for( int j=0; j<tempJJ.cols; j++ ) {	
				//		if( j%3==0 ) cout << endl; 
				//		std::cout << std::setw(14) << std::scientific << tempJJ.at<double>(i, j) << "  ";
				//	}
				//	cout << endl; 
				//}
				//cout << endl;

				//cout << endl << "tempJJ2 = " << endl;
				//for( int i=0; i<tempJJ2.rows; i++ ) {
				//	for( int j=0; j<tempJJ2.cols; j++ ) {	
				//		if( j%3==0 ) cout << endl; 
				//		std::cout << std::setw(14) << std::scientific << tempJJ2.at<double>(i, j) << "  ";
				//	}
				//	cout << endl; 
				//}
				//cout << endl;
				
				//cout << endl << "JJ = " << endl;
				//for( int i=0; i<JJ.rows; i++ ) {
				//	for( int j=0; j<JJ.cols; j++ ) {	
				//		if( j%3==0 ) cout << endl; 
				//		std::cout << std::setw(14) << std::scientific << JJ.at<double>(i, j) << "  ";
				//	}
				//	cout << endl; 
				//}
				//cout << endl;

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
		//Sleep(500);
		


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
			cout << "- ";
			energy_before = new_energy; 
			lambda *= 0.51; 
		} else {
			cout << "+ ";
			for( int label=0; label < lines.size(); label++ ) {
				for( int i=0; i < numOfParametersPerLine; i++ ) {
					const double& delta = X.at<double>( label * numOfParametersPerLine + i ); 
					lines[label]->updateParameterWithDelta( i, -delta ); 
				}
			}
			lambda *= 2.13; 
		}

		
		// update the end points of the line
		vector<float> minT( (int) labelings.size(), (std::numeric_limits<float>::max)() );
		vector<float> maxT( (int) labelings.size(), (std::numeric_limits<float>::min)() );
		for( int site = 0; site < dataPoints.size(); site++ ) { // For each data point
			int label = labelings[site]; 
			Vec3f p1, p2;
			lines[label]->getEndPoints( p1, p2 ); 

			Vec3f& pos = p1;
			Vec3f dir = p2 - p1; 
			dir /= sqrt( dir.dot( dir ) ); // normalize the direction 
			float t = ( Vec3f(dataPoints[site]) - pos ).dot( dir );
			maxT[label] = max( t+1, maxT[label] );
			minT[label] = min( t-1, minT[label] );
		}
		for( int label=0; label<labelings.size(); label++ ) {
			if( minT[label] < maxT[label] ) {
				Vec3f p1, p2;
				lines[label]->getEndPoints( p1, p2 ); 

				Vec3f& pos = p1;
				Vec3f dir = p2 - p1; 
				dir /= sqrt( dir.dot( dir ) ); // normalize the direction 
				
				lines[label]->setPositions( pos + dir * minT[label], pos + dir * maxT[label] ); 
			}
		}
	}
}
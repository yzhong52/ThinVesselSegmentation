#include "GLLineModel.h"

using namespace GLViewer;



GLLineModel::GLLineModel( cv::Vec3i size ) : size( size ), render_mode(1) {
	// creat a mutex 
	hMutex = CreateMutex( NULL, false, NULL );
}

GLLineModel::~GLLineModel( void ){
	// destroy the mutex 
	CloseHandle( hMutex ); 
}


void GLLineModel::render(void){
	// in case there is any previously bind texture, you need to unbind them
	glBindTexture( GL_TEXTURE_3D, NULL );

	///////////////////////////////////////////////////
	//// Draw the axis
	///////////////////////////////////////////////////
	//	glBegin( GL_LINES );
	//	// x-axis
	//	glColor3f(  1.0f, 0.0f, 0.0f ); 
	//	glVertex3i( 0, 0, 0 ); 
	//	glVertex3i( size[0], 0, 0 ); 
	//	// y-axis
	//	glColor3f(  0.0f, 1.0f, 0.0f ); 
	//	glVertex3i( 0, 0, 0 ); 
	//	glVertex3i( 0, size[1], 0 ); 
	//	// z-axis
	//	glColor3f(  0.0f, 0.0f, 1.0f ); 
	//	glVertex3i( 0, 0, 0 ); 
	//	glVertex3i( 0, 0, size[2] ); 
	//	glEnd();

	WaitForSingleObject( hMutex, INFINITE );

	/////////////////////////////////////////////////
	// draw the data points 
	/////////////////////////////////////////////////
	if( render_mode & 4 ){
		glPointSize( 3.0 );
		glBegin( GL_POINTS );
		for( int i=0; i < (int) dataPoints.size(); i++ ) {
			const int& lineID = labelings[i]; 
			Vec3f prj = lines[lineID]->projection( dataPoints[i] ); 
			glColor3ubv( &lineColors[lineID][0] ); 
			glVertex3fv( &prj[0] ); 
			// drawing data points
			//glColor3f( 0.1f, 0.1f, 0.1f ); 
			//glVertex3iv( &dataPoints[i][0] ); 
		} 
		glEnd();
	}

	/////////////////////////////////////////////////
	// draw the end points of the lines
	/////////////////////////////////////////////////
	//glPointSize( 6.0 );
	//glBegin( GL_POINTS );
	//for( int i=0; i < (int) dataPoints.size(); i++ ) {
	//	int lineID = labelings[i]; 
	//	Vec3f p1, p2; 
	//	lines[lineID]->getEndPoints( p1, p2 ); 
	//	glColor3ubv( &lineColors[lineID][0] ); 
	//	glVertex3fv( &p1[0] ); 
	//	glVertex3fv( &p2[0] ); 
	//} 
	//glEnd();

	if( render_mode & 1 ) {
		glBegin( GL_LINES );
		for( int i=0; i < (int) dataPoints.size(); i++ ) {
			// label id
			const int lineID = labelings[i]; 
			// actual position
			Vec3f prj = lines[lineID]->projection( dataPoints[i] ); 

			/////////////////////////////////////////////////
			// draw a short line alond the line model
			/////////////////////////////////////////////////
			glColor3f( 0.3f, 0.3f, 0.3f ); 
			// direction of the line
			Vec3f dir = lines[lineID]->getDirection(); 
			glVertex3fv( &(prj + dir * 0.5 )[0] ); 
			glVertex3fv( &(prj - dir * 0.5 )[0] ); 
		} 
		glEnd();
	}

	if( render_mode & 2 ) {
		glBegin( GL_LINES );
		for( int i=0; i < (int) dataPoints.size(); i++ ) {
			// label id
			const int lineID = labelings[i]; 
			// actual position
			Vec3f prj = lines[lineID]->projection( dataPoints[i] ); 

			/////////////////////////////////////////////////
			// draw the lines of the projection direction
			/////////////////////////////////////////////////
			// projection point 
			glColor3ubv( &lineColors[lineID][0] );  glVertex3fv( &prj[0] ); 
			// data points
			glColor3f( 0.1f, 0.1f, 0.1f );          glVertex3iv( &dataPoints[i][0] ); 
		} 
		glEnd();
	}

	ReleaseMutex( hMutex );
	
}


void GLLineModel::updatePoints( const vector<Vec3i>& pts ){
	WaitForSingleObject( hMutex, INFINITE );
	dataPoints = pts; 
	ReleaseMutex( hMutex );
}

void GLLineModel::updateModel( const vector<Line3D*>& lns, const vector<int>& lbls )
{
	WaitForSingleObject( hMutex, INFINITE );
	if( lbls.size()==dataPoints.size() ) {
		lines = lns; 
		labelings = lbls; 
		int num = (int) lns.size() - (int) lineColors.size();
		for( int i=0; i<num; i++ ) {
			Vec3b c( 
				(rand()%228 ) + 28,
				(rand()%228 ) + 28,
				(rand()%228 ) + 28 ); 
			lineColors.push_back( c ); 
		}
	} else {
		cout << "Error: Update model fail." << endl; 
		std::wcout << "  Location: file "<< _CRT_WIDE(__FILE__) << ", line " << __LINE__ << std::endl; 
		system( "pause" );
	}
	ReleaseMutex( hMutex );
}

void GLLineModel::init(void){
	glDisable(GL_DEPTH_TEST);
	glEnable(GL_BLEND);
	glBlendFunc(GL_ONE, GL_ONE);
	glBlendEquation( GL_MAX_EXT ); 
	//cout << "Volumn Rendeing Mode is set to MIP" << endl;

	//// Antialiasing
	glDisable(GL_LINE_SMOOTH);
	glHint (GL_LINE_SMOOTH_HINT, GL_FASTEST );

	//glEnable( GL_POLYGON_SMOOTH_HINT );
	//glHint (GL_POLYGON_SMOOTH_HINT, GL_NICEST);
	//glHint (GL_PERSPECTIVE_CORRECTION_HINT, GL_NICEST);

	//glDisable(GL_DEPTH_TEST);
	//glEnable(GL_BLEND);
	//glBlendFunc(GL_ONE, GL_ONE);
	//glBlendEquation( GL_MAX_EXT ); 

	
	
	glEnable( GL_POINT_SPRITE ); // GL_POINT_SPRITE_ARB if you're
                                 // using the functionality as an extension.

    glEnable( GL_POINT_SMOOTH );
    glEnable( GL_BLEND );
    glBlendFunc( GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA );
    
}

void GLLineModel::keyboard( unsigned char key ){
	render_mode++; 
	if( render_mode==0 ) render_mode = 1; 
}
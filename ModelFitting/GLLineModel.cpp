#include "GLLineModel.h"

using namespace GLViewer;



GLLineModel::GLLineModel( cv::Vec3i size ) : size( size ) {
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

	glColor3f( 1.0f, 1.0f, 1.0f ); 
	glColor3f( 1.0f, 0.0f, 0.0f ); 
	
	WaitForSingleObject( hMutex, INFINITE );
	glBegin( GL_POINTS );
	for( int i=0; i < (int) dataPoints.size(); i++ ) {
		glVertex3iv( &dataPoints[i][0] ); 
	} 
	glEnd();
	glBegin( GL_LINES );
	for( int i=0; i< (int) lines.size(); i++ ) {
		Vec3f pos1 = lines[i].getPos(); 
		Vec3f pos2 = lines[i].getDir() + lines[i].getPos(); 
		glVertex3fv( &pos1[0] ); 
		glVertex3fv( &pos2[0] ); 
		cv::Vec3f dir; 
	}
	glEnd();
	ReleaseMutex( hMutex );
	
}


void GLLineModel::updatePoints( const vector<Vec3i>& pts ){
	WaitForSingleObject( hMutex, INFINITE );
	dataPoints = pts; 
	ReleaseMutex( hMutex );
}

void GLLineModel::updateLines( const vector<Line3D>& lns ){
	WaitForSingleObject( hMutex, INFINITE );
	lines = lns; 
	ReleaseMutex( hMutex );
}

void GLLineModel::updatelabelings( const vector<int>& lbls ){

}

void GLLineModel::init(void){
	glDisable(GL_DEPTH_TEST);
	glEnable(GL_BLEND);
	glBlendFunc(GL_ONE, GL_ONE);
	glBlendEquation( GL_MAX_EXT ); 
	//cout << "Volumn Rendeing Mode is set to MIP" << endl;

	//// Antialiasing
	//glEnable (GL_LINE_SMOOTH);
	//glHint (GL_LINE_SMOOTH_HINT, GL_NICEST );

	//glEnable( GL_POLYGON_SMOOTH_HINT );
	//glHint (GL_POLYGON_SMOOTH_HINT, GL_NICEST);
	//glHint (GL_PERSPECTIVE_CORRECTION_HINT, GL_NICEST);

	//glDisable(GL_DEPTH_TEST);
	//glEnable(GL_BLEND);
	//glBlendFunc(GL_ONE, GL_ONE);
	//glBlendEquation( GL_MAX_EXT ); 
}
#include "GLCamera.h"
#include <math.h>

void rotate_axis( 
	float u, float v, float w,        /*Axis*/
	float x, float y, float z,        /*The Point That We Want to Roate */
	float& nx, float& ny, float& nz,  /*Result*/
	float degree ) 
{
	// change from degree to radian
	float A = degree * 3.14159265f / 180.0f; 
	float c = cos(A);
	float s = sin(A);
	float C = 1.0f - c;

	if( abs(c) > 0.999 ) {
		nx = x;
		ny = y;
		nz = z;
	}

	// Reference: http://en.wikipedia.org/wiki/Rotation_matrix#Rotation_matrix_from_axis_and_angle
	float Q[3][3];
	Q[0][0] = u * u * C + c;
	Q[0][1] = v * u * C + w * s;
	Q[0][2] = w * u * C - v * s;

	Q[1][0] = v * u * C - w * s;
	Q[1][1] = v * v * C + c;
	Q[1][2] = w * v * C + u * s;

	Q[2][0] = u * w * C + v * s;
	Q[2][1] = w * v * C - u * s;
	Q[2][2] = w * w * C + c;

	nx = x * Q[0][0] + y * Q[1][0] + z * Q[2][0];
	ny = x * Q[0][1] + y * Q[1][1] + z * Q[2][1];
	nz = x * Q[0][2] + y * Q[1][2] + z * Q[2][2];
}

GLCamera::GLCamera(void)
	: scale(1.0f)
{
	navigationMode = None;

	// Rotation
	xrot = 0;              
	yrot = 0;
	rotate_speed = 0.001f;

	// rotation axis
	vec_y[0] = 0;
	vec_y[1] = 1;
	vec_y[2] = 0;
	vec_x[0] = 1;
	vec_x[1] = 0;
	vec_x[2] = 0;

	// translation
	t[0] = t[1] = t[2] = 0;
	translate_speed = 0.2f;

	elapsedTick = 0; 
}


GLCamera::~GLCamera(void)
{
}



void GLCamera::zoomIn(void){
	translate_speed /= 1.01f; 
	rotate_speed  /= 1.01f; 
	scale *= 1.01f; 
}

void GLCamera::zoomOut(void){
	translate_speed /= 0.99f; 
	rotate_speed  /= 0.99f; 
	scale *= 0.99f; 
}


void GLCamera::rotate_scene(void){
	static int tick = GetTickCount();
	static int oldtick = GetTickCount(); 
	tick = GetTickCount(); 
	elapsedTick = tick - oldtick;
	oldtick = tick; 

	// update the rotation matrix as well as the rotation axis
	glRotatef( xrot * elapsedTick, vec_y[0], vec_y[1], vec_y[2] );
	rotate_axis( vec_y[0], vec_y[1], vec_y[2], 
                 vec_x[0], vec_x[1], vec_x[2],
		         vec_x[0], vec_x[1], vec_x[2], -xrot * elapsedTick );
	glRotatef( yrot * elapsedTick, vec_x[0], vec_x[1], vec_x[2] );
	rotate_axis( vec_x[0], vec_x[1], vec_x[2], 
		         vec_y[0], vec_y[1], vec_y[2],
		         vec_y[0], vec_y[1], vec_y[2], -yrot * elapsedTick );
}
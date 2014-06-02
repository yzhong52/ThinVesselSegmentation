#include "GLCamera.h"
#include <cmath>
#include <ctime>
#include <iostream>
using namespace std;


#if _MSC_VER && !__INTEL_COMPILER
#include <Windows.h>
#else
#include <sys/time.h>
#endif // _MSC_VER && !__INTEL_COMPILER

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

    if( std::abs(c) > 0.999 )
    {
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

GLCamera::GLCamera(const float& rotate_speed)
    : rot_x( 0 ), rot_y( 0 ), rot_speed( rotate_speed )
    , scale( 1 )
{
    navigationMode = None;

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



void GLCamera::zoomIn(void)
{
    translate_speed /= 1.01f;
    // rotate_speed  *= 1.01f;
    scale *= 1.01f;
}

void GLCamera::zoomOut(void)
{
    translate_speed /= 0.99f;
    // rotate_speed  *= 0.99f;
    scale *= 0.99f;
}


void GLCamera::rotate_scene(void)
{
    double elapsed = 0.0;

#if _MSC_VER && !__INTEL_COMPILER // if Windows OS
    LARGE_INTEGER frequency;        // ticks per second
    static LARGE_INTEGER t1, t2;    // ticks
    // get ticks per second
    QueryPerformanceFrequency(&frequency);
    // get current time
    QueryPerformanceCounter(&t2);
    static bool flag = true;
    if( flag ) {
        t1 = t2; // t1 is initialized only once here
        flag = false;
    }
    // compute and print the elapsed time in millisec
    elapsed = 1000.0*(t2.QuadPart-t1.QuadPart)/frequency.QuadPart;
    t1 = t2;
#else // if Linux/Mac OS
    static timeval t1, t2;
    // get current time
    gettimeofday(&t2, NULL);
    static bool flag = true;
    if( flag ) {
        t1 = t2; // t1 is initialized only once here
        flag = false;
    }
    // compute and print the elapsed time in millisec
    elapsed = (t2.tv_sec - t1.tv_sec) * 1000.0; // sec to ms
    elapsed += (t2.tv_usec - t1.tv_usec) / 1000.0;     // us to ms
    t1 = t2;
#endif

    // update the rotation matrix as well as the rotation axis
    rotate_x( rot_y * elapsed );
    rotate_y( rot_x * elapsed );
}


void GLCamera::rotate_x( float degree )
{
    // update the rotation matrix as well as the rotation axis
    glRotatef( degree, vec_x[0], vec_x[1], vec_x[2] );
    rotate_axis( vec_x[0], vec_x[1], vec_x[2], // IN: Roation axis
                 vec_y[0], vec_y[1], vec_y[2], // IN
                 vec_y[0], vec_y[1], vec_y[2], // OUT
                 -degree );
}


void GLCamera::rotate_y( float degree )
{
    // update the rotation matrix as well as the rotation axis
    glRotatef( degree, vec_y[0], vec_y[1], vec_y[2] );
    rotate_axis( vec_y[0], vec_y[1], vec_y[2], // IN: Roation axis
                 vec_x[0], vec_x[1], vec_x[2], // IN
                 vec_x[0], vec_x[1], vec_x[2], // OUT
                 -degree );
}

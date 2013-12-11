///////////////////////////////////////////////////////////////////////////////
// ViewFormGL.cpp
// ==============
// View component of OpenGL dialog window
//
//  AUTHORL Song Ho Ahn (song.ahn@gmail.com)
// CREATED: 2006-07-10
// UPDATED: 2006-08-15
///////////////////////////////////////////////////////////////////////////////

#include "ViewFormGL.h"
#include "resource.h"
#include "Log.h"
using namespace Win;

///////////////////////////////////////////////////////////////////////////////
// default ctor
///////////////////////////////////////////////////////////////////////////////
ViewFormGL::ViewFormGL()
{
}


///////////////////////////////////////////////////////////////////////////////
// default dtor
///////////////////////////////////////////////////////////////////////////////
ViewFormGL::~ViewFormGL()
{
}



///////////////////////////////////////////////////////////////////////////////
// initialize all controls
///////////////////////////////////////////////////////////////////////////////
void ViewFormGL::initControls(HWND handle)
{
    // set all controls
    buttonAnimate.set(handle, IDC_ANIMATE);
    radioFill.set(handle, IDC_FILL);
    radioWireframe.set(handle, IDC_WIREFRAME);
    radioPoint.set(handle, IDC_POINT);
    trackbarRed.set(handle, IDC_RED);
    trackbarGreen.set(handle, IDC_GREEN);
    trackbarBlue.set(handle, IDC_BLUE);

    // initial state
    radioFill.check();
    trackbarRed.setRange(0, 255);
    trackbarRed.setPos(0);
    trackbarGreen.setRange(0, 255);
    trackbarGreen.setPos(0);
    trackbarBlue.setRange(0, 255);
    trackbarBlue.setPos(0);
}



///////////////////////////////////////////////////////////////////////////////
// update caption of animate button
///////////////////////////////////////////////////////////////////////////////
void ViewFormGL::animate(bool flag)
{
    if(flag)
        buttonAnimate.setText(L"Stop");
    else
        buttonAnimate.setText(L"Animate");
}



///////////////////////////////////////////////////////////////////////////////
// update trackbars
///////////////////////////////////////////////////////////////////////////////
void ViewFormGL::updateTrackbars(HWND handle, int position)
{
    if(handle == trackbarRed.getHandle())
    {
        trackbarRed.setPos(position);
    }
    else if(handle == trackbarGreen.getHandle())
    {
        trackbarGreen.setPos(position);
    }
    else if(handle == trackbarBlue.getHandle())
    {
        trackbarBlue.setPos(position);
    }
}

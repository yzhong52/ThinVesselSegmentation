///////////////////////////////////////////////////////////////////////////////
// ViewGL.cpp
// ==========
// View component of OpenGL window
//
//  AUTHORL Song Ho Ahn (song.ahn@gmail.com)
// CREATED: 2006-07-10
// UPDATED: 2006-07-10
///////////////////////////////////////////////////////////////////////////////

#include "ViewGL.h"
#include "resource.h"
using namespace Win;

///////////////////////////////////////////////////////////////////////////////
// default ctor
///////////////////////////////////////////////////////////////////////////////
ViewGL::ViewGL() : hdc(0), hglrc(0)
{
}


///////////////////////////////////////////////////////////////////////////////
// default dtor
///////////////////////////////////////////////////////////////////////////////
ViewGL::~ViewGL()
{
}



///////////////////////////////////////////////////////////////////////////////
//
///////////////////////////////////////////////////////////////////////////////
void ViewGL::closeContext(HWND handle)
{
    if(!hdc || !hglrc)
        return;

    // delete DC and RC
    ::wglMakeCurrent(0, 0);
    ::wglDeleteContext(hglrc);
    ::ReleaseDC(handle, hdc);

    hdc = 0;
    hglrc = 0;
}



///////////////////////////////////////////////////////////////////////////////
// create OpenGL rendering context
///////////////////////////////////////////////////////////////////////////////
bool ViewGL::createContext(HWND handle, int colorBits, int depthBits, int stencilBits)
{
    // retrieve a handle to a display device context
    hdc = ::GetDC(handle);

    // set pixel format
    if(!setPixelFormat(hdc, colorBits, depthBits, stencilBits))
    {
        ::MessageBox(0, L"Cannot set a suitable pixel format.", L"Error", MB_ICONEXCLAMATION | MB_OK);
        ::ReleaseDC(handle, hdc);                     // remove device context
        return false;
    }

    // create a new OpenGL rendering context
    hglrc = ::wglCreateContext(hdc);
    //::wglMakeCurrent(hdc, hglrc);

    ::ReleaseDC(handle, hdc);
    return true;
}



///////////////////////////////////////////////////////////////////////////////
// choose pixel format
// By default, pdf.dwFlags is set PFD_DRAW_TO_WINDOW, PFD_DOUBLEBUFFER and PFD_SUPPORT_OPENGL.
///////////////////////////////////////////////////////////////////////////////
bool ViewGL::setPixelFormat(HDC hdc, int colorBits, int depthBits, int stencilBits)
{
    PIXELFORMATDESCRIPTOR pfd;

    // find out the best matched pixel format
    int pixelFormat = findPixelFormat(hdc, colorBits, depthBits, stencilBits);
    if(pixelFormat == 0)
        return false;

    // set members of PIXELFORMATDESCRIPTOR with given mode ID
    ::DescribePixelFormat(hdc, pixelFormat, sizeof(pfd), &pfd);

    // set the fixel format
    if(!::SetPixelFormat(hdc, pixelFormat, &pfd))
        return false;

    return true;
}



///////////////////////////////////////////////////////////////////////////////
// find the best pixel format
///////////////////////////////////////////////////////////////////////////////
int ViewGL::findPixelFormat(HDC hdc, int colorBits, int depthBits, int stencilBits)
{
    int currMode;                               // pixel format mode ID
    int bestMode = 0;                           // return value, best pixel format
    int currScore = 0;                          // points of current mode
    int bestScore = 0;                          // points of best candidate
    PIXELFORMATDESCRIPTOR pfd;

    // search the available formats for the best mode
    bestMode = 0;
    bestScore = 0;
    for(currMode = 1; ::DescribePixelFormat(hdc, currMode, sizeof(pfd), &pfd) > 0; ++currMode)
    {
        // ignore if cannot support opengl
        if(!(pfd.dwFlags & PFD_SUPPORT_OPENGL))
            continue;

        // ignore if cannot render into a window
        if(!(pfd.dwFlags & PFD_DRAW_TO_WINDOW))
            continue;

        // ignore if cannot support rgba mode
        if((pfd.iPixelType != PFD_TYPE_RGBA) || (pfd.dwFlags & PFD_NEED_PALETTE))
            continue;

        // ignore if not double buffer
        if(!(pfd.dwFlags & PFD_DOUBLEBUFFER))
            continue;

        // try to find best candidate
        currScore = 0;

        // colour bits
        if(pfd.cColorBits >= colorBits) ++currScore;
        if(pfd.cColorBits == colorBits) ++currScore;

        // depth bits
        if(pfd.cDepthBits >= depthBits) ++currScore;
        if(pfd.cDepthBits == depthBits) ++currScore;

        // stencil bits
        if(pfd.cStencilBits >= stencilBits) ++currScore;
        if(pfd.cStencilBits == stencilBits) ++currScore;

        // alpha bits
        if(pfd.cAlphaBits > 0) ++currScore;

        // check if it is best mode so far
        if(currScore > bestScore)
        {
            bestScore = currScore;
            bestMode = currMode;
        }
    }

    return bestMode;
}



///////////////////////////////////////////////////////////////////////////////
// swap OpenGL frame buffers
///////////////////////////////////////////////////////////////////////////////
void ViewGL::swapBuffers()
{
    ::SwapBuffers(hdc);
}




///////////////////////////////////////////////////////////////////////////////
// ControllerMain.cpp
// ==================
// Derived Controller class for main window
//
//  AUTHOR: Song Ho Ahn (song.ahn@gmail.com)
// CREATED: 2006-07-09
// UPDATED: 2013-11-28
///////////////////////////////////////////////////////////////////////////////

#include <windows.h>
#include <commctrl.h>                   // common controls
#include <sstream>
#include "ControllerMain.h"
#include "resource.h"

using namespace Win;


// handle events(messages) on all child windows that belong to the parent window.
// For example, close all child windows when the parent got WM_CLOSE message.
// lParam can be used to specify a event or message.
bool CALLBACK enumerateChildren(HWND childHandle, LPARAM lParam);



ControllerMain::ControllerMain() : glHandle(0), formHandle(0)
{
}



int ControllerMain::command(int id, int cmd, LPARAM msg)
{
    switch(id)
    {
    case ID_FILE_EXIT:
        ::PostMessage(handle, WM_CLOSE, 0, 0);
        break;

    }

    return 0;
}



int ControllerMain::close()
{
    // close all child windows first
    ::EnumChildWindows(handle, (WNDENUMPROC)enumerateChildren, (LPARAM)WM_CLOSE);

    ::DestroyWindow(handle);    // close itself
    return 0;
}



int ControllerMain::destroy()
{
    ::PostQuitMessage(0);       // exit the message loop
    return 0;
}



int ControllerMain::create()
{
    return 0;
}



int ControllerMain::size(int w, int h, WPARAM wParam)
{
    RECT rect;

    // get client dimension of mainWin
    ::GetClientRect(handle, &rect);
    int mainClientWidth = rect.right - rect.left;
    int mainClientHeight = rect.bottom - rect.top;

    // get height of status bar
    HWND statusHandle = ::GetDlgItem(handle, IDC_STATUSBAR);
    ::GetWindowRect(statusHandle, &rect);
    int statusHeight = rect.bottom - rect.top;

    // get height of glDialog
    ::GetWindowRect(formHandle, &rect);
    int formHeight = rect.bottom - rect.top;

    // resize the height of glWin and reposition glDialog & status bar
    int glHeight = mainClientHeight - formHeight - statusHeight;
    ::SetWindowPos(glHandle, 0, 0, 0, mainClientWidth, glHeight, SWP_NOZORDER);
    ::SetWindowPos(formHandle, 0, 0, glHeight, mainClientWidth, formHeight, SWP_NOZORDER);
    ::InvalidateRect(formHandle, 0, TRUE);      // force to repaint
    ::SendMessage(statusHandle, WM_SIZE, 0, 0); // automatically resize width, so send 0s
    ::InvalidateRect(statusHandle, 0, FALSE);   // force to repaint

    // display OpenGL window dimension on the status bar
    std::wstringstream wss;
    wss << "OpenGL Window Size: " << mainClientWidth << " x " << glHeight;
    ::SendMessage(statusHandle, SB_SETTEXT, 0, (LPARAM)wss.str().c_str());

    return 0;
}



///////////////////////////////////////////////////////////////////////////////
// enumerate all child windows
///////////////////////////////////////////////////////////////////////////////
bool CALLBACK enumerateChildren(HWND handle, LPARAM lParam)
{
    if(lParam == WM_CLOSE)
    {
        ::SendMessage(handle, WM_CLOSE, 0, 0);      // close child windows
    }

    return true;
}

///////////////////////////////////////////////////////////////////////////////
// main.cpp
// ========
// main driver
//
//  AUTHOR: Song Ho Ahn (song.ahn@gmail.com)
// CREATED: 2006-06-28
// UPDATED: 2013-11-27
///////////////////////////////////////////////////////////////////////////////

#define WIN32_LEAN_AND_MEAN             // exclude rarely-used stuff from Windows headers

#include <windows.h>
#include <commctrl.h>                   // common controls
#include "Window.h"
#include "DialogWindow.h"
#include "ControllerMain.h"
#include "ControllerGL.h"
#include "ControllerFormGL.h"
#include "ModelGL.h"
#include "ViewGL.h"
#include "ViewFormGL.h"
#include "resource.h"
#include "Log.h"


// function declarations
int mainMessageLoop(HACCEL hAccelTable=0);




///////////////////////////////////////////////////////////////////////////////
// main function of a windows application
///////////////////////////////////////////////////////////////////////////////
int WINAPI WinMain(HINSTANCE hInst, HINSTANCE hPrevInst, LPSTR cmdArgs, int cmdShow)
{
    Win::logMode(Win::LOG_MODE_BOTH);

    // register slider(trackbar) from comctl32.dll brfore creating windows
    INITCOMMONCONTROLSEX commonCtrls;
    commonCtrls.dwSize = sizeof(commonCtrls);
    commonCtrls.dwICC = ICC_BAR_CLASSES;        // trackbar is in this class
    ::InitCommonControlsEx(&commonCtrls);

    // get app name from resource file
    wchar_t name[256];
    ::LoadString(hInst, IDS_APP_NAME, name, 256);

    Win::ControllerMain mainCtrl;
    Win::Window mainWin(hInst, name, 0, &mainCtrl);

    // add menu to window class
    mainWin.setMenuName(MAKEINTRESOURCE(IDR_MAIN_MENU));
    mainWin.setWidth(500);
    mainWin.setHeight(660);
    mainWin.setWindowStyleEx(WS_EX_WINDOWEDGE);

    // create a window and show
    if( mainWin.create() )
        Win::log("Main window is created.");
    else
        Win::log("[ERROR] Failed to create main window.");

    // create model and view components for controller
    ModelGL modelGL;
    Win::ViewGL viewGL;
    Win::ViewFormGL viewFormGL;

    // create OpenGL rendering window as a child
    Win::ControllerGL glCtrl(&modelGL, &viewGL);
    Win::Window glWin(hInst, L"WindowGL", mainWin.getHandle(), &glCtrl);
    glWin.setClassStyle(CS_OWNDC);
    glWin.setWindowStyle(WS_CHILD | WS_VISIBLE | WS_CLIPSIBLINGS | WS_CLIPCHILDREN);
    glWin.setWidth(500);
    glWin.setHeight(500);
    if(glWin.create())
        Win::log("OpenGL child window is created.");
    else
        Win::log("[ERROR] Failed to create OpenGL window.");

    // create a child dialog box contains controls
    Win::ControllerFormGL formCtrl(&modelGL, &viewFormGL);
    Win::DialogWindow glDialog(hInst, IDD_CONTROLS, mainWin.getHandle(), &formCtrl);
    glDialog.setWidth(500);
    glDialog.setHeight(160);
    if(glDialog.create())
        Win::log("OpenGL form dialog is created.");
    else
        Win::log("[ERROR] Failed to create OpenGL form dialog.");

    // create status bar window using CreateWindowEx()
    // mainWin must pass WM_SIZE message to the status bar
    // So, mainWin accesses the status bar with GetDlgItem(handle, IDC_STATUSBAR).
    HWND statusHandle = ::CreateWindowEx(0, STATUSCLASSNAME, 0, WS_CHILD | WS_VISIBLE | SBARS_SIZEGRIP,
                                         0, 0, 0, 0, mainWin.getHandle(), (HMENU)IDC_STATUSBAR, ::GetModuleHandle(0), 0);
    if(statusHandle)
        Win::log("Status bar window is created.");
    else
        Win::log("[ERROR] Failed to create status bar window.");
    ::SendMessage(statusHandle, SB_SETTEXT, 0, (LPARAM)L"Ready");

    // send window handles to mainCtrl, they are used for resizing window
    mainCtrl.setGLHandle(glWin.getHandle());
    mainCtrl.setFormHandle(glDialog.getHandle());

    // place the opengl form dialog in right place, bottome of the opengl rendering window
    ::SetWindowPos(glDialog.getHandle(), 0, 0, 300, 300, 160, SWP_NOZORDER);

    // compute height of all sub-windows
    int height = 0;
    RECT rect;
    ::GetWindowRect(glWin.getHandle(), &rect);      // get size of glWin
    height += rect.bottom - rect.top;
    ::GetWindowRect(glDialog.getHandle(), &rect);   // get size of glDialog
    height += rect.bottom - rect.top;
    ::GetWindowRect(statusHandle, &rect);           // get size of status bar
    height += rect.bottom - rect.top;

    // resize main window, so all sub windows are fit into the client area of main window
    DWORD style = ::GetWindowLong(mainWin.getHandle(), GWL_STYLE);      // get current window style
    DWORD styleEx = ::GetWindowLong(mainWin.getHandle(), GWL_EXSTYLE);  // get current extended window style
    rect.left = 0;
    rect.right = 500;
    rect.top = 0;
    rect.bottom = height;
    ::AdjustWindowRectEx(&rect, style, TRUE, styleEx);
    ::SetWindowPos(mainWin.getHandle(), 0, 100, 100, rect.right-rect.left, rect.bottom-rect.top, SWP_NOZORDER);

    // show all windows
    glWin.show();
    glDialog.show();
    mainWin.show();
    //::SendMessage(mainWin.getHandle(), WM_NCPAINT, 1, 0);   // repaint window frame

    // main message loop //////////////////////////////////////////////////////
    int exitCode;
    HACCEL hAccelTable = 0;
    //hAccelTable = ::LoadAccelerators(hInst, MAKEINTRESOURCE(ID_ACCEL));
    exitCode = mainMessageLoop(hAccelTable);

    return exitCode;
}



///////////////////////////////////////////////////////////////////////////////
// main message loop
///////////////////////////////////////////////////////////////////////////////
int mainMessageLoop(HACCEL hAccelTable)
{
    HWND activeHandle;
    MSG msg;

    while(::GetMessage(&msg, 0, 0, 0) > 0)  // loop until WM_QUIT(0) received
    {
        // determine the activated window is dialog box
        // skip if messages are for the dialog windows
        activeHandle = GetActiveWindow();
        if(::GetWindowLongPtr(activeHandle, GWL_EXSTYLE) & WS_EX_CONTROLPARENT) // WS_EX_CONTROLPARENT is automatically added by CreateDialogBox()
        {
            if(::IsDialogMessage(activeHandle, &msg))
                continue;   // message handled, back to while-loop
        }

        // now, handle window messages
        if(!::TranslateAccelerator(msg.hwnd, hAccelTable, &msg))
        {
            ::TranslateMessage(&msg);
            ::DispatchMessage(&msg);
        }
    }

    return (int)msg.wParam;                 // return nExitCode of PostQuitMessage()
}

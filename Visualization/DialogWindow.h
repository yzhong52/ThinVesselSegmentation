///////////////////////////////////////////////////////////////////////////////
// DialogWindow.h
// ==============
// A class of Dialog Box for MS Windows
// It creates a dialog box with CreateDialogParam() API call.
//
//  AUTHOR: Song Ho Ahn
// CREATED: 2005-03-16
// UPDATED: 2006-06-20
///////////////////////////////////////////////////////////////////////////////

#ifndef WIN_DIALOG_WINDOW_H
#define WIN_DIALOG_WINDOW_H

#include <windows.h>
#include "Controller.h"


namespace Win
{
    class DialogWindow
    {
    public:
        // ctor/dtor with minimal args
        // The second param, id should be the template ID of a dialog in resource.rc(IDD_*).
        DialogWindow(HINSTANCE hInst, WORD id, HWND hParent, Controller* ctrl);
        ~DialogWindow();

        HWND create();                          // create a dialog
        void show(int cmdShow=SW_SHOWDEFAULT);  // make the window visible or invisible
        HWND getHandle()                        { return handle; }; // return dialog window handle

        // setters
        void setPosition(int x, int y)          { this->x = x; this->y = y; };
        void setWidth(int w)                    { width = w; };
        void setHeight(int h)                   { height = h; };


    private:
        HWND handle;                            // handle to this window
        HWND parentHandle;                      // handle to parent window
        WORD id;
        int x;                                  // window position X
        int y;                                  // window position Y
        int width;                              // window width
        int height;                             // window height
        HINSTANCE instance;                     // handle to instance
        Win::Controller *controller;            // pointer to controller
    };
}

#endif

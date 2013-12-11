///////////////////////////////////////////////////////////////////////////////
// procedure.cpp
// =============
// Window procedure and dialog procedure callback functions.
// Windows will call this function whenever an event is triggered. It routes
// the message to the controller associated with window handle.
//
//  AUTHOR: Song Ho Ahn (song.ahn@gmail.com)
// CREATED: 2006-06-24
// UPDATED: 2012-08-27
///////////////////////////////////////////////////////////////////////////////

#include "procedure.h"
#include "Controller.h"

///////////////////////////////////////////////////////////////////////////////
// Window Procedure
///////////////////////////////////////////////////////////////////////////////
LRESULT CALLBACK Win::windowProcedure(HWND hwnd, UINT msg, WPARAM wParam, LPARAM lParam)
{
    LRESULT returnValue = 0;        // return value

    // find controller associated with window handle
    static Win::Controller *ctrl;
    ctrl = (Controller*)::GetWindowLongPtr(hwnd, GWL_USERDATA);

    if(msg == WM_NCCREATE)  // Non-Client Create
    {
        // WM_NCCREATE message is called before non-client parts(border,
        // titlebar, menu,etc) are created. This message comes with a pointer
        // to CREATESTRUCT in lParam. The lpCreateParams member of CREATESTRUCT
        // actually contains the value of lpPraram of CreateWindowEX().
        // First, retrieve the pointrer to the controller specified when
        // Win::Window is setup.
        ctrl = (Controller*)(((CREATESTRUCT*)lParam)->lpCreateParams);
        ctrl->setHandle(hwnd);

        // Second, store the pointer to the Controller into GWL_USERDATA,
        // so, other messege can be routed to the associated Controller.
        ::SetWindowLongPtr(hwnd, GWL_USERDATA, (LONG_PTR)ctrl);

        return ::DefWindowProc(hwnd, msg, wParam, lParam);
    }

    // check NULL pointer, because GWL_USERDATA is initially 0, and
    // we store a valid pointer value when WM_NCCREATE is called.
    if(!ctrl)
        return ::DefWindowProc(hwnd, msg, wParam, lParam);

    // route messages to the associated controller
    switch(msg)
    {
    case WM_CREATE:
        returnValue = ctrl->create();
        break;

    case WM_SIZE:
        returnValue = ctrl->size(LOWORD(lParam), HIWORD(lParam), (int)wParam);    // width, height, type
        break;

    case WM_ENABLE:
        {
        bool flag;
        if(wParam) flag = true;
        else flag = false;
        returnValue = ctrl->enable(flag);  // TRUE or FALSE
        break;
        }

    case WM_PAINT:
        ctrl->paint();
        returnValue = ::DefWindowProc(hwnd, msg, wParam, lParam);
        break;

    case WM_COMMAND:
        returnValue = ctrl->command(LOWORD(wParam), HIWORD(wParam), lParam);   // id, code, msg
        break;

    case WM_MOUSEMOVE:
        returnValue = ctrl->mouseMove(wParam, LOWORD(lParam), HIWORD(lParam));  // state, x, y
        break;

    case WM_CLOSE:
        returnValue = ctrl->close();
        break;

    case WM_DESTROY:
        returnValue = ctrl->destroy();
        break;

    case WM_SYSCOMMAND:
        //returnValue = ctrl->sysCommand(wParam, lParam);
        returnValue = ::DefWindowProc(hwnd, msg, wParam, lParam);
        break;
/*
        // Disable Alt-F4 and screensavers
        switch (wparam & 0xfff0)
        {
        case SC_CLOSE:
        case SC_SCREENSAVE:
            ret = 0;
            break;

        default:
            returnValue = DefWindowProc(hwnd, message, wparam, lparam);
            break;
        }
        break;
*/

    case WM_CHAR:
        //returnValue = ctrl->char(wParam, lParam);   // route keycode
        break;

    case WM_KEYDOWN:
    case WM_SYSKEYDOWN:
        returnValue = ctrl->keyDown((int)wParam, lParam);                       // keyCode, keyDetail
        break;

    case WM_KEYUP:
    case WM_SYSKEYUP:
        returnValue = ctrl->keyUp((int)wParam, lParam);                         // keyCode, keyDetail
        break;

    case WM_LBUTTONDOWN:
        returnValue = ctrl->lButtonDown(wParam, LOWORD(lParam), HIWORD(lParam)); // state, x, y
        break;

    case WM_LBUTTONUP:
        returnValue = ctrl->lButtonUp(wParam, LOWORD(lParam), HIWORD(lParam));   // state, x, y
        break;

    case WM_RBUTTONDOWN:
        returnValue = ctrl->rButtonDown(wParam, LOWORD(lParam), HIWORD(lParam)); // state, x, y
        break;

    case WM_RBUTTONUP:
        returnValue = ctrl->rButtonUp(wParam, LOWORD(lParam), HIWORD(lParam));   // state, x, y
        break;

    case WM_MBUTTONDOWN:
        returnValue = ctrl->mButtonDown(wParam, LOWORD(lParam), HIWORD(lParam)); // state, x, y
        break;

    case WM_MBUTTONUP:
        returnValue = ctrl->mButtonUp(wParam, LOWORD(lParam), HIWORD(lParam));   // state, x, y
        break;

    case WM_MOUSEWHEEL:
        returnValue = ctrl->mouseWheel((short)LOWORD(wParam), (short)HIWORD(wParam)/WHEEL_DELTA, (short)LOWORD(lParam), (short)HIWORD(lParam));   // state, delta, x, y
        break;

    case WM_HSCROLL:
        returnValue = ctrl->hScroll(wParam, lParam);
        break;

    case WM_VSCROLL:
        returnValue = ctrl->vScroll(wParam, lParam);
        break;

    case WM_TIMER:
        returnValue = ctrl->timer(LOWORD(wParam), HIWORD(wParam));
        break;

    case WM_NOTIFY:
        returnValue = ctrl->notify((int)wParam, lParam);                        // controllerID, lParam
        break;

    case WM_CONTEXTMENU:
        returnValue = ctrl->contextMenu((HWND)wParam, LOWORD(lParam), HIWORD(lParam));    // handle, x, y (from screen coords)

    //case WM_ERASEBKGND:
    //    returnValue = ctrl->eraseBkgnd((HDC)wParam);                            // handle to device context
    //    break;

    default:
        returnValue = ::DefWindowProc(hwnd, msg, wParam, lParam);
    }

    return returnValue;
}



///////////////////////////////////////////////////////////////////////////////
// Dialog Procedure
// It must return true if the message is handled.
///////////////////////////////////////////////////////////////////////////////
BOOL CALLBACK Win::dialogProcedure(HWND hwnd, UINT msg, WPARAM wParam, LPARAM lParam)
{
    // find controller associated with window handle
    static Win::Controller *ctrl;
    ctrl = (Controller*)::GetWindowLongPtr(hwnd, GWL_USERDATA);

    // WM_INITDIALOG message is called before displaying the dialog box.
    // lParam contains the value of dwInitParam of CreateDialogBoxParam(),
    // which is the pointer to the Controller.
    if(msg == WM_INITDIALOG)
    {
        // First, retrieve the pointrer to the controller associated with
        // the current dialog box.
        ctrl = (Controller*)lParam;
        ctrl->setHandle(hwnd);

        // Second, store the pointer to the Controller into GWL_USERDATA,
        // so, other messege can be routed to the associated Controller.
        ::SetWindowLongPtr(hwnd, GWL_USERDATA, (LONG_PTR)ctrl);

        // When WM_INITDIALOG is called, all controls in the dialog are created.
        // It is good time to initalize the appearance of controls here.
        // NOTE that we don't handle WM_CREATE message for dialogs because
        // the message is sent before controls have been create, so you cannot
        // access them in WM_CREATE message handler.
        ctrl->create();

        return true;
    }

    // check NULL pointer, because GWL_USERDATA is initially 0, and
    // we store a valid pointer value when WM_NCCREATE is called.
    if(!ctrl)
        return false;

    switch(msg)
    {
    case WM_COMMAND:
        ctrl->command(LOWORD(wParam), HIWORD(wParam), lParam);   // id, code, msg
        return true;

    case WM_TIMER:
        ctrl->timer(LOWORD(wParam), HIWORD(wParam));
        return true;

    case WM_PAINT:
        ctrl->paint();
        ::DefWindowProc(hwnd, msg, wParam, lParam);
        return true;

    case WM_DESTROY:
        ctrl->destroy();
        return true;

    case WM_CLOSE:
        ctrl->close();
        return true;

    case WM_HSCROLL:
        ctrl->hScroll(wParam, lParam);
        return true;

    case WM_VSCROLL:
        ctrl->vScroll(wParam, lParam);
        return true;

    case WM_NOTIFY:
        ctrl->notify((int)wParam, lParam);                      // controllerID, lParam
        return true;

    case WM_MOUSEMOVE:
        ctrl->mouseMove(wParam, LOWORD(lParam), HIWORD(lParam));
        //ctrl->mouseMove(wParam, (int)GET_X_LPARAM(lParam), (int)GET_Y_LPARAM(lParam));  // state, x, y
        return true;

    case WM_LBUTTONUP:
        ctrl->lButtonUp(wParam, LOWORD(lParam), HIWORD(lParam));    // state, x, y
        return true;

    case WM_CONTEXTMENU:
        ctrl->contextMenu((HWND)wParam, LOWORD(lParam), HIWORD(lParam));    // handle, x, y (from screen coords)
        return true;

    default:
        return false;
    }
}

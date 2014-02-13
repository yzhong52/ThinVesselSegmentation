///////////////////////////////////////////////////////////////////////////////
// ControllerFormGL.cpp
// ====================
// Derived Controller class for OpenGL dialog window
//
//  AUTHOR: Song Ho Ahn (song.ahn@gamil.com)
// CREATED: 2006-07-09
// UPDATED: 2006-08-15
///////////////////////////////////////////////////////////////////////////////

#include <process.h>                                // for _beginthreadex()
#include "ControllerFormGL.h"
#include "resource.h"
#include "Log.h"
using namespace Win;



///////////////////////////////////////////////////////////////////////////////
// default contructor
///////////////////////////////////////////////////////////////////////////////
ControllerFormGL::ControllerFormGL(ModelGL* model, ViewFormGL* view) : model(model), view(view)
{
}



///////////////////////////////////////////////////////////////////////////////
// handle WM_CREATE
///////////////////////////////////////////////////////////////////////////////
int ControllerFormGL::create()
{
    // initialize all controls
    view->initControls(handle);

    // place the opengl form dialog in right place, bottome of the opengl rendering window
    //RECT rect = {0, 0, 4, 8};
    //::MapDialogRect(glDialog.getHandle(), &rect);
    //int width = MulDiv(300, 4, rect.right);
    //::SetWindowPos(handle, HWND_TOP, 0, 300, 300, 200, 0);

    return 0;
}



///////////////////////////////////////////////////////////////////////////////
// handle WM_COMMAND
///////////////////////////////////////////////////////////////////////////////
int ControllerFormGL::command(int id, int command, LPARAM msg)
{
    static bool flag = false;

    switch(id)
    {
    case IDC_ANIMATE:
        if(command == BN_CLICKED)
        {
            flag = !flag;
            // model->animate(flag);
            // view->animate(flag);
        }
        break;

    case IDC_FILL:
        if(command == BN_CLICKED)
        {
            // model->setDrawMode(0);
        }
        break;

    case IDC_WIREFRAME:
        if(command == BN_CLICKED)
        {
            // model->setDrawMode(1);
        }
        break;

    case IDC_POINT:
        if(command == BN_CLICKED)
        {
            // model->setDrawMode(2);
        }
        break;
    }

    return 0;
}



///////////////////////////////////////////////////////////////////////////////
// handle horizontal scroll notification
///////////////////////////////////////////////////////////////////////////////
int ControllerFormGL::hScroll(WPARAM wParam, LPARAM lParam)
{
    // check if the message comming from trackbar
    HWND trackbarHandle = (HWND)lParam;

    int position = HIWORD(wParam);              // current tick mark position
	int temp; 
    if(trackbarHandle)
    {
        // get control ID
        int trackbarId = ::GetDlgCtrlID(trackbarHandle);

        switch(LOWORD(wParam))
        {
		case TB_THUMBTRACK:     // user dragged the slider
		case WM_HSCROLL:
            position = SendMessage(trackbarHandle, TBM_GETPOS, 0, 0);
            if( trackbarId == IDC_MIN ) {
				temp = view->getTrackbarWindowCenterMax();
				if( temp < position ) {
					view->updateTrackbarWindowCenterMax( position );
					model->updateWindowCenter( position, position ); 
				} else {
					model->updateWindowCenterMin( position );
				}
			}
            else if( trackbarId == IDC_MAX ) {
				temp = view->getTrackbarWindowCenterMin();
				if( temp > position ) {
					view->updateTrackbarWindowCenterMin( position );
					model->updateWindowCenter( position, position ); 
				} else {
					model->updateWindowCenterMax( position );
				}
			}
            break;
        }
    }

    return 0;
}




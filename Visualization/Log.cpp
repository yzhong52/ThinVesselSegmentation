///////////////////////////////////////////////////////////////////////////////
// Log.cpp
// =======
// It prints out any log messages to file or dialog box.
// Log class is a singleton class which is contructed by calling
// Log::getInstance() (lazy initialization), and is destructed automatically
// when the application is terminated.
//
// In order to log, use Win::log() function with appropriate formats.
// For example, Win::log(L"My number: %d\n", 123).
// It is similar to printf() function of C standard libirary.
//
// The template of the log dialog window is defined in log.rc and logResource.h
// You must include both resource file with this source codes.
// The dialog window cannot be closed by user once it is created. But it will be
// destroyed when the application terminated.
//
//  AUTHOR: Song Ho Ahn (song.ahn@gmail.com)
// CREATED: 2006-07-14
// UPDATED: 2008-05-07
///////////////////////////////////////////////////////////////////////////////

#include <cstdarg>
#include <cwchar>
#include <sstream>
#include <iomanip>
#include "Log.h"
#include "logResource.h"                            // for log dialog resource
#include "wcharUtil.h"
using namespace Win;


const char* LOG_FILE = "log.txt";

BOOL CALLBACK logDialogProc(HWND hwnd, UINT message, WPARAM wParam, LPARAM lParam);



///////////////////////////////////////////////////////////////////////////////
// constructor
///////////////////////////////////////////////////////////////////////////////
Log::Log() : logMode(LOG_MODE_FILE), dialogHandle(0), listHandle(0)
{
    // open log file
    logFile.open(LOG_FILE, std::ios::out);
    if(logFile.fail())
        return;

    // first put starting date and time
    logFile << L"===== Log started at "
            << getDate() << L", "
            << getTime() << L". =====\n\n"
            << std::flush;
}



///////////////////////////////////////////////////////////////////////////////
// destructor
///////////////////////////////////////////////////////////////////////////////
Log::~Log()
{
    // close opened file
    logFile << L"\n\n===== END OF LOG =====\n";
    logFile.close();

    // destroy dilalog
    if(dialogHandle)
    {
        ::DestroyWindow(dialogHandle);
        dialogHandle = 0;
    }
}



///////////////////////////////////////////////////////////////////////////////
// instantiate a singleton instance if not exist
///////////////////////////////////////////////////////////////////////////////
Log& Log::getInstance()
{
    static Log self;
    return self;
}



///////////////////////////////////////////////////////////////////////////////
// add message to log
///////////////////////////////////////////////////////////////////////////////
void Log::put(const std::wstring& message)
{
    if(logMode != LOG_MODE_FILE)
    {
        std::wstring str;
        str = getTime() + L": " + message;
        //long index = ::SendMessage(listHandle, LB_ADDSTRING, 0, (LPARAM)str.c_str());
        //::SendMessage(listHandle, LB_SETTOPINDEX, index, 0);  // set focus to current line

        // SendMessage() in worker thread may cause deadlock, use SendMessageTimeOut() instead
        long index;
        LRESULT result = ::SendMessageTimeout(listHandle, LB_ADDSTRING, 0, (LPARAM)str.c_str(), SMTO_NORMAL | SMTO_ABORTIFHUNG, 500, (PDWORD_PTR)&index);
        if(result) // non-zero means succeeded
            ::SendMessageTimeout(listHandle, LB_SETTOPINDEX, index, 0, SMTO_NORMAL, 500, 0);  // set focus to current line
    }

    if(logMode != LOG_MODE_DIALOG)
    {
        // put time first and append message
        logFile << getTime() << L"  "
                << message
                << L"\n"
                << std::flush;
    }
}




///////////////////////////////////////////////////////////////////////////////
// get system date as a string
///////////////////////////////////////////////////////////////////////////////
const std::wstring Log::getDate()
{
    std::wstringstream wss;

    SYSTEMTIME sysTime;
    ::GetLocalTime(&sysTime);

    wss << std::setfill(L'0');
    wss << sysTime.wYear << L"-" << std::setw(2)
        << sysTime.wMonth << L"-" << std::setw(2)
        << sysTime.wDay;

    return wss.str();
}



///////////////////////////////////////////////////////////////////////////////
// get system time as a string
///////////////////////////////////////////////////////////////////////////////
const std::wstring Log::getTime()
{
    std::wstringstream wss;

    SYSTEMTIME sysTime;
    ::GetLocalTime(&sysTime);

    wss << std::setfill(L'0');
    wss << sysTime.wHour << L":" << std::setw(2)
        << sysTime.wMinute << L":" << std::setw(2)
        << sysTime.wSecond;

    return wss.str();
}



///////////////////////////////////////////////////////////////////////////////
// switch logging target; file or dialog window
///////////////////////////////////////////////////////////////////////////////
void Log::setMode(int mode)
{
    if(mode > LOG_MODE_BOTH) return;                // invalid mode number

    if(logMode == LOG_MODE_FILE && mode == LOG_MODE_DIALOG)
    {
        logFile << getTime() << L"  "
                << L"Redirect log to dialog box.\n"
                << std::flush;
    }

    logMode = mode;

    if(logMode != LOG_MODE_FILE)                    // to dialog
    {
        if(!dialogHandle)
        {
            // create log dialog if it is not created yet
            dialogHandle = ::CreateDialog(0,                        // handle to instance
                                          MAKEINTRESOURCE(IDD_LOG), // id of dialog template
                                          0,                        // handle to parent window
                                          (DLGPROC)logDialogProc);  // pointer to procedure function
            ::ShowWindow(dialogHandle, SW_SHOW);
            ::UpdateWindow(dialogHandle);

            // store handle to listbox
            listHandle = ::GetDlgItem(dialogHandle, IDC_LIST_LOG);

            // set horizontal extent to display the horizontal scroll bar in the listbox
            ::SendMessage(listHandle, LB_SETHORIZONTALEXTENT, 1000, 0);

            // positioning the dialog at the bottom of screen
            RECT rect1, rect2;
            int x, y;
            ::SystemParametersInfo(SPI_GETWORKAREA, 0, &rect1, 0);  // get workarea excluding taskbar
            ::GetWindowRect(dialogHandle, &rect2);

            // compute the position of dialog (lower-right corner)
            x = rect1.right - (rect2.right - rect2.left);
            y = rect1.bottom - (rect2.bottom - rect2.top);

            // reposition dialog window at lower-right corder of screen
            ::SetWindowPos(dialogHandle, HWND_TOP, x, y, 0, 0, SWP_NOSIZE);
        }
    }
    else
    {
        // minimize log dialog if it is shown
        if(dialogHandle)
            ::ShowWindow(dialogHandle, SW_MINIMIZE);
    }
}



///////////////////////////////////////////////////////////////////////////////
// C-style printf fuction
///////////////////////////////////////////////////////////////////////////////
void Win::log(const wchar_t *format, ...)
{
    wchar_t buffer[LOG_MAX_STRING];

    // do the formating
    va_list valist;
    va_start(valist, format);
    _vsnwprintf(buffer, LOG_MAX_STRING, format, valist);
    va_end(valist);

    Log::getInstance().put(buffer);
}



void Win::log(const char *format, ...)
{
    char buffer[LOG_MAX_STRING];

    // do the formating
    va_list valist;
    va_start(valist, format);
    _vsnprintf(buffer, LOG_MAX_STRING, format, valist);
    va_end(valist);

    Log::getInstance().put(toWchar(buffer));
}



void Win::log(const std::wstring& str)
{
    Log::getInstance().put(str);
}



///////////////////////////////////////////////////////////////////////////////
// set logging target
///////////////////////////////////////////////////////////////////////////////
void Win::logMode(int mode)
{
    Log::getInstance().setMode(mode);
}



///////////////////////////////////////////////////////////////////////////////
// process log dialog messages
///////////////////////////////////////////////////////////////////////////////
BOOL CALLBACK logDialogProc(HWND hwnd, UINT message, WPARAM wParam, LPARAM lParam)
{
    switch(message)
    {
        case WM_INITDIALOG:
            return true;

        case WM_COMMAND:
        {
            int id = LOWORD(wParam);
            int cmd = HIWORD(wParam);
            switch(id)
            {
            case IDC_BUTTON_MINIMIZE:
                if(cmd == BN_CLICKED)
                {
                    // minimize
                    ::ShowWindow(hwnd, SW_MINIMIZE);
                }
                break;

            case IDC_BUTTON_CLEAR:
                if(cmd == BN_CLICKED)
                {
                    // clear log list
                    ::SendMessage(::GetDlgItem(hwnd, IDC_LIST_LOG), LB_RESETCONTENT, 0, 0);
                }
                break;
            }
            return true;
        }

        case WM_TIMER:
            return true;

        case WM_DESTROY:
            ::PostQuitMessage(0);
            return true;

        case WM_CLOSE:
            ::DestroyWindow(hwnd);
            return true;
    }

    return false;
}

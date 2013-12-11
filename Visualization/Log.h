///////////////////////////////////////////////////////////////////////////////
// Log.h
// =====
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
// UPDATED: 2006-07-24
///////////////////////////////////////////////////////////////////////////////

#ifndef WIN_LOG_H
#define WIN_LOG_H

#include <string>
#include <fstream>
#include <windows.h>

namespace Win
{
    enum { LOG_MODE_FILE = 0, LOG_MODE_DIALOG, LOG_MODE_BOTH }; // log output selection
    enum { LOG_MAX_STRING = 1024 };

    // Clients are actually use this functions to send log messages.
    // USAGE: Win::log("I am the number %d.", 1);
    void log(const std::wstring& str);
    void log(const wchar_t *format, ...);
    void log(const char *format, ...);
    extern void logMode(int mode);



    // singleton class ////////////////////////////////////////////////////////
    class Log
    {
    public:
        ~Log();

        static Log& getInstance();              // return reference to this class object

        void setMode(int mode);                 // set log target: file or dialog
        void put(const std::wstring& str);      // print log message

    private:
        Log();                                  // hide it here to prevent instantiating this class
        Log(const Log& rhs);                    // must no body for copy ctor, so this class cannot have copy ctor

        const std::wstring getTime();           // return system time as string
        const std::wstring getDate();           // return system date as string

        int logMode;                            // file, dialog or both
        std::wofstream logFile;                 // log file handle
        HWND dialogHandle;                      // handle to dialog window
        HWND listHandle;                        // handle to listbox
    };
    ///////////////////////////////////////////////////////////////////////////
}

#endif


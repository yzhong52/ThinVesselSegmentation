#ifndef EXAMPLE3_H_INCLUDED
#define EXAMPLE3_H_INCLUDED

#include <python3.2/Python.h>

void send_email()
{
    Py_Initialize();
    PyRun_SimpleString("import sys");
    PyRun_SimpleString("sys.path.append('./')");

    PyObject *pModule = NULL;
    pModule = PyImport_ImportModule( "send_email" );

    PyObject *pFunc = NULL;
    pFunc = PyObject_GetAttrString(pModule, "send_an_email");

    PyEval_CallObject(pFunc, NULL);

    Py_Initialize();
}


#endif // EXAMPLE3_H_INCLUDED

#pragma once

#include <string>
#include <opencv2/core/core.hpp>

template <class T> struct TypeInfo
{
    static std::string str()
    {
        return "undefined";
    }
    static int CV_TYPE()
    {
        return -1;
    }
};

template <> struct TypeInfo <int>
{
    static std::string str()
    {
        return "int";
    }
    static int CV_TYPE()
    {
        return CV_32S;
    }
};

template <> struct TypeInfo <short>
{
    static std::string str()
    {
        return "short";
    }
    static int CV_TYPE()
    {
        return CV_16S;
    }
};

template <> struct TypeInfo <float>
{
    static std::string str()
    {
        return "float";
    }
    static int CV_TYPE()
    {
        return CV_32F;
    }
};

template <> struct TypeInfo <double>
{
    static std::string str()
    {
        return "double";
    }
    static int CV_TYPE()
    {
        return CV_64F;
    }
};

template <> struct TypeInfo <unsigned short>
{
    static std::string str()
    {
        return "unsigned short";
    }
    static int CV_TYPE()
    {
        return CV_16U;
    }
};

template <> struct TypeInfo <unsigned char>
{
    static std::string str()
    {
        return "unsigned char";
    }
    static int CV_TYPE()
    {
        return CV_8U;
    }
};

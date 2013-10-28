

/* this ALWAYS GENERATED file contains the IIDs and CLSIDs */

/* link this file in with the server and any clients */


 /* File created by MIDL compiler version 6.00.0366 */
/* at Sat Jan 17 20:05:40 2009
 */
/* Compiler settings for win32\mwcomutil.idl:
    Oicf, W1, Zp8, env=Win32 (32b run)
    protocol : dce , ms_ext, c_ext, robust
    error checks: allocation ref bounds_check enum stub_data 
    VC __declspec() decoration level: 
         __declspec(uuid()), __declspec(selectany), __declspec(novtable)
         DECLSPEC_UUID(), MIDL_INTERFACE()
*/
//@@MIDL_FILE_HEADING(  )

#pragma warning( disable: 4049 )  /* more than 64k source lines */


#ifdef __cplusplus
extern "C"{
#endif 


#include <rpc.h>
#include <rpcndr.h>

#ifdef _MIDL_USE_GUIDDEF_

#ifndef INITGUID
#define INITGUID
#include <guiddef.h>
#undef INITGUID
#else
#include <guiddef.h>
#endif

#define MIDL_DEFINE_GUID(type,name,l,w1,w2,b1,b2,b3,b4,b5,b6,b7,b8) \
        DEFINE_GUID(name,l,w1,w2,b1,b2,b3,b4,b5,b6,b7,b8)

#else // !_MIDL_USE_GUIDDEF_

#ifndef __IID_DEFINED__
#define __IID_DEFINED__

typedef struct _IID
{
    unsigned long x;
    unsigned short s1;
    unsigned short s2;
    unsigned char  c[8];
} IID;

#endif // __IID_DEFINED__

#ifndef CLSID_DEFINED
#define CLSID_DEFINED
typedef IID CLSID;
#endif // CLSID_DEFINED

#define MIDL_DEFINE_GUID(type,name,l,w1,w2,b1,b2,b3,b4,b5,b6,b7,b8) \
        const type name = {l,w1,w2,{b1,b2,b3,b4,b5,b6,b7,b8}}

#endif !_MIDL_USE_GUIDDEF_

MIDL_DEFINE_GUID(IID, IID_IMWUtil,0xC47EA90E,0x56D1,0x11d5,0xB1,0x59,0x00,0xD0,0xB7,0xBA,0x75,0x44);


MIDL_DEFINE_GUID(IID, LIBID_MWComUtil,0xB127C57D,0xED57,0x4BB9,0xB9,0xE0,0xD6,0x10,0x84,0xA3,0x7F,0x6B);


MIDL_DEFINE_GUID(CLSID, CLSID_MWField,0x0A145DE2,0x5EED,0x4C66,0x94,0x43,0x48,0x72,0x04,0x72,0xCE,0xEC);


MIDL_DEFINE_GUID(CLSID, CLSID_MWStruct,0xE72CDF78,0xC3E2,0x41FA,0x89,0xA1,0x00,0x4A,0x25,0x37,0x1D,0xA5);


MIDL_DEFINE_GUID(CLSID, CLSID_MWComplex,0xE53E3B4B,0xE24C,0x449E,0x91,0xAC,0x2E,0xE1,0xB8,0x16,0xA8,0x31);


MIDL_DEFINE_GUID(CLSID, CLSID_MWSparse,0x3E347B31,0x48C3,0x427F,0xBE,0x41,0x71,0xE7,0xBC,0x77,0x33,0x3A);


MIDL_DEFINE_GUID(CLSID, CLSID_MWArg,0x6C7F426D,0xA7DE,0x4EA6,0x87,0x15,0xA8,0x8E,0x6C,0x2F,0x73,0x5C);


MIDL_DEFINE_GUID(CLSID, CLSID_MWArrayFormatFlags,0xA0CFC286,0x9018,0x411D,0x83,0x17,0xA2,0x0F,0x16,0x60,0x4F,0x5F);


MIDL_DEFINE_GUID(CLSID, CLSID_MWDataConversionFlags,0xA79AD773,0x380B,0x451C,0x9D,0xF1,0x85,0x35,0x5C,0x3E,0x40,0xBD);


MIDL_DEFINE_GUID(CLSID, CLSID_MWUtil,0xCE30EF50,0xDB22,0x4934,0xB4,0xC2,0xC0,0x8E,0xF2,0x05,0xEE,0x55);


MIDL_DEFINE_GUID(CLSID, CLSID_MWFlags,0xE31161EC,0x1E33,0x458C,0x8C,0xAD,0x14,0x15,0x16,0x5E,0x3F,0x53);

#undef MIDL_DEFINE_GUID

#ifdef __cplusplus
}
#endif




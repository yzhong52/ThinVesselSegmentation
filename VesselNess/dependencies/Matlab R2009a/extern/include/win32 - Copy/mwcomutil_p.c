

/* this ALWAYS GENERATED file contains the proxy stub code */


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

#if !defined(_M_IA64) && !defined(_M_AMD64)


#pragma warning( disable: 4049 )  /* more than 64k source lines */
#if _MSC_VER >= 1200
#pragma warning(push)
#endif
#pragma warning( disable: 4100 ) /* unreferenced arguments in x86 call */
#pragma warning( disable: 4211 )  /* redefine extent to static */
#pragma warning( disable: 4232 )  /* dllimport identity*/
#pragma optimize("", off ) 

#define USE_STUBLESS_PROXY


/* verify that the <rpcproxy.h> version is high enough to compile this file*/
#ifndef __REDQ_RPCPROXY_H_VERSION__
#define __REQUIRED_RPCPROXY_H_VERSION__ 475
#endif


#include "rpcproxy.h"
#ifndef __RPCPROXY_H_VERSION__
#error this stub requires an updated version of <rpcproxy.h>
#endif // __RPCPROXY_H_VERSION__


#include "mwcomutil.h"

#define TYPE_FORMAT_STRING_SIZE   1057                              
#define PROC_FORMAT_STRING_SIZE   541                               
#define TRANSMIT_AS_TABLE_SIZE    0            
#define WIRE_MARSHAL_TABLE_SIZE   1            

typedef struct _MIDL_TYPE_FORMAT_STRING
    {
    short          Pad;
    unsigned char  Format[ TYPE_FORMAT_STRING_SIZE ];
    } MIDL_TYPE_FORMAT_STRING;

typedef struct _MIDL_PROC_FORMAT_STRING
    {
    short          Pad;
    unsigned char  Format[ PROC_FORMAT_STRING_SIZE ];
    } MIDL_PROC_FORMAT_STRING;


static RPC_SYNTAX_IDENTIFIER  _RpcTransferSyntax = 
{{0x8A885D04,0x1CEB,0x11C9,{0x9F,0xE8,0x08,0x00,0x2B,0x10,0x48,0x60}},{2,0}};


extern const MIDL_TYPE_FORMAT_STRING __MIDL_TypeFormatString;
extern const MIDL_PROC_FORMAT_STRING __MIDL_ProcFormatString;


extern const MIDL_STUB_DESC Object_StubDesc;


extern const MIDL_SERVER_INFO IMWUtil_ServerInfo;
extern const MIDL_STUBLESS_PROXY_INFO IMWUtil_ProxyInfo;


extern const USER_MARSHAL_ROUTINE_QUADRUPLE UserMarshalRoutines[ WIRE_MARSHAL_TABLE_SIZE ];

#if !defined(__RPC_WIN32__)
#error  Invalid build platform for this stub.
#endif

#if !(TARGET_IS_NT50_OR_LATER)
#error You need a Windows 2000 or later to run this stub because it uses these features:
#error   /robust command line switch.
#error However, your C/C++ compilation flags indicate you intend to run this app on earlier systems.
#error This app will die there with the RPC_X_WRONG_STUB_VERSION error.
#endif


static const MIDL_PROC_FORMAT_STRING __MIDL_ProcFormatString =
    {
        0,
        {

	/* Procedure MWPack */

			0x33,		/* FC_AUTO_HANDLE */
			0x6c,		/* Old Flags:  object, Oi2 */
/*  2 */	NdrFcLong( 0x0 ),	/* 0 */
/*  6 */	NdrFcShort( 0x7 ),	/* 7 */
/*  8 */	NdrFcShort( 0x20c ),	/* x86 Stack size/offset = 524 */
/* 10 */	NdrFcShort( 0x0 ),	/* 0 */
/* 12 */	NdrFcShort( 0x8 ),	/* 8 */
/* 14 */	0x47,		/* Oi2 Flags:  srv must size, clt must size, has return, has ext, */
			0x22,		/* 34 */
/* 16 */	0x8,		/* 8 */
			0x7,		/* Ext Flags:  new corr desc, clt corr check, srv corr check, */
/* 18 */	NdrFcShort( 0x20 ),	/* 32 */
/* 20 */	NdrFcShort( 0x400 ),	/* 1024 */
/* 22 */	NdrFcShort( 0x0 ),	/* 0 */

	/* Parameter pVarArg */

/* 24 */	NdrFcShort( 0x4113 ),	/* Flags:  must size, must free, out, simple ref, srv alloc size=16 */
/* 26 */	NdrFcShort( 0x4 ),	/* x86 Stack size/offset = 4 */
/* 28 */	NdrFcShort( 0x404 ),	/* Type Offset=1028 */

	/* Parameter Var0 */

/* 30 */	NdrFcShort( 0x8b ),	/* Flags:  must size, must free, in, by val, */
/* 32 */	NdrFcShort( 0x8 ),	/* x86 Stack size/offset = 8 */
/* 34 */	NdrFcShort( 0x412 ),	/* Type Offset=1042 */

	/* Parameter Var1 */

/* 36 */	NdrFcShort( 0x8b ),	/* Flags:  must size, must free, in, by val, */
/* 38 */	NdrFcShort( 0x18 ),	/* x86 Stack size/offset = 24 */
/* 40 */	NdrFcShort( 0x412 ),	/* Type Offset=1042 */

	/* Parameter Var2 */

/* 42 */	NdrFcShort( 0x8b ),	/* Flags:  must size, must free, in, by val, */
/* 44 */	NdrFcShort( 0x28 ),	/* x86 Stack size/offset = 40 */
/* 46 */	NdrFcShort( 0x412 ),	/* Type Offset=1042 */

	/* Parameter Var3 */

/* 48 */	NdrFcShort( 0x8b ),	/* Flags:  must size, must free, in, by val, */
/* 50 */	NdrFcShort( 0x38 ),	/* x86 Stack size/offset = 56 */
/* 52 */	NdrFcShort( 0x412 ),	/* Type Offset=1042 */

	/* Parameter Var4 */

/* 54 */	NdrFcShort( 0x8b ),	/* Flags:  must size, must free, in, by val, */
/* 56 */	NdrFcShort( 0x48 ),	/* x86 Stack size/offset = 72 */
/* 58 */	NdrFcShort( 0x412 ),	/* Type Offset=1042 */

	/* Parameter Var5 */

/* 60 */	NdrFcShort( 0x8b ),	/* Flags:  must size, must free, in, by val, */
/* 62 */	NdrFcShort( 0x58 ),	/* x86 Stack size/offset = 88 */
/* 64 */	NdrFcShort( 0x412 ),	/* Type Offset=1042 */

	/* Parameter Var6 */

/* 66 */	NdrFcShort( 0x8b ),	/* Flags:  must size, must free, in, by val, */
/* 68 */	NdrFcShort( 0x68 ),	/* x86 Stack size/offset = 104 */
/* 70 */	NdrFcShort( 0x412 ),	/* Type Offset=1042 */

	/* Parameter Var7 */

/* 72 */	NdrFcShort( 0x8b ),	/* Flags:  must size, must free, in, by val, */
/* 74 */	NdrFcShort( 0x78 ),	/* x86 Stack size/offset = 120 */
/* 76 */	NdrFcShort( 0x412 ),	/* Type Offset=1042 */

	/* Parameter Var8 */

/* 78 */	NdrFcShort( 0x8b ),	/* Flags:  must size, must free, in, by val, */
/* 80 */	NdrFcShort( 0x88 ),	/* x86 Stack size/offset = 136 */
/* 82 */	NdrFcShort( 0x412 ),	/* Type Offset=1042 */

	/* Parameter Var9 */

/* 84 */	NdrFcShort( 0x8b ),	/* Flags:  must size, must free, in, by val, */
/* 86 */	NdrFcShort( 0x98 ),	/* x86 Stack size/offset = 152 */
/* 88 */	NdrFcShort( 0x412 ),	/* Type Offset=1042 */

	/* Parameter Var10 */

/* 90 */	NdrFcShort( 0x8b ),	/* Flags:  must size, must free, in, by val, */
/* 92 */	NdrFcShort( 0xa8 ),	/* x86 Stack size/offset = 168 */
/* 94 */	NdrFcShort( 0x412 ),	/* Type Offset=1042 */

	/* Parameter Var11 */

/* 96 */	NdrFcShort( 0x8b ),	/* Flags:  must size, must free, in, by val, */
/* 98 */	NdrFcShort( 0xb8 ),	/* x86 Stack size/offset = 184 */
/* 100 */	NdrFcShort( 0x412 ),	/* Type Offset=1042 */

	/* Parameter Var12 */

/* 102 */	NdrFcShort( 0x8b ),	/* Flags:  must size, must free, in, by val, */
/* 104 */	NdrFcShort( 0xc8 ),	/* x86 Stack size/offset = 200 */
/* 106 */	NdrFcShort( 0x412 ),	/* Type Offset=1042 */

	/* Parameter Var13 */

/* 108 */	NdrFcShort( 0x8b ),	/* Flags:  must size, must free, in, by val, */
/* 110 */	NdrFcShort( 0xd8 ),	/* x86 Stack size/offset = 216 */
/* 112 */	NdrFcShort( 0x412 ),	/* Type Offset=1042 */

	/* Parameter Var14 */

/* 114 */	NdrFcShort( 0x8b ),	/* Flags:  must size, must free, in, by val, */
/* 116 */	NdrFcShort( 0xe8 ),	/* x86 Stack size/offset = 232 */
/* 118 */	NdrFcShort( 0x412 ),	/* Type Offset=1042 */

	/* Parameter Var15 */

/* 120 */	NdrFcShort( 0x8b ),	/* Flags:  must size, must free, in, by val, */
/* 122 */	NdrFcShort( 0xf8 ),	/* x86 Stack size/offset = 248 */
/* 124 */	NdrFcShort( 0x412 ),	/* Type Offset=1042 */

	/* Parameter Var16 */

/* 126 */	NdrFcShort( 0x8b ),	/* Flags:  must size, must free, in, by val, */
/* 128 */	NdrFcShort( 0x108 ),	/* x86 Stack size/offset = 264 */
/* 130 */	NdrFcShort( 0x412 ),	/* Type Offset=1042 */

	/* Parameter Var17 */

/* 132 */	NdrFcShort( 0x8b ),	/* Flags:  must size, must free, in, by val, */
/* 134 */	NdrFcShort( 0x118 ),	/* x86 Stack size/offset = 280 */
/* 136 */	NdrFcShort( 0x412 ),	/* Type Offset=1042 */

	/* Parameter Var18 */

/* 138 */	NdrFcShort( 0x8b ),	/* Flags:  must size, must free, in, by val, */
/* 140 */	NdrFcShort( 0x128 ),	/* x86 Stack size/offset = 296 */
/* 142 */	NdrFcShort( 0x412 ),	/* Type Offset=1042 */

	/* Parameter Var19 */

/* 144 */	NdrFcShort( 0x8b ),	/* Flags:  must size, must free, in, by val, */
/* 146 */	NdrFcShort( 0x138 ),	/* x86 Stack size/offset = 312 */
/* 148 */	NdrFcShort( 0x412 ),	/* Type Offset=1042 */

	/* Parameter Var20 */

/* 150 */	NdrFcShort( 0x8b ),	/* Flags:  must size, must free, in, by val, */
/* 152 */	NdrFcShort( 0x148 ),	/* x86 Stack size/offset = 328 */
/* 154 */	NdrFcShort( 0x412 ),	/* Type Offset=1042 */

	/* Parameter Var21 */

/* 156 */	NdrFcShort( 0x8b ),	/* Flags:  must size, must free, in, by val, */
/* 158 */	NdrFcShort( 0x158 ),	/* x86 Stack size/offset = 344 */
/* 160 */	NdrFcShort( 0x412 ),	/* Type Offset=1042 */

	/* Parameter Var22 */

/* 162 */	NdrFcShort( 0x8b ),	/* Flags:  must size, must free, in, by val, */
/* 164 */	NdrFcShort( 0x168 ),	/* x86 Stack size/offset = 360 */
/* 166 */	NdrFcShort( 0x412 ),	/* Type Offset=1042 */

	/* Parameter Var23 */

/* 168 */	NdrFcShort( 0x8b ),	/* Flags:  must size, must free, in, by val, */
/* 170 */	NdrFcShort( 0x178 ),	/* x86 Stack size/offset = 376 */
/* 172 */	NdrFcShort( 0x412 ),	/* Type Offset=1042 */

	/* Parameter Var24 */

/* 174 */	NdrFcShort( 0x8b ),	/* Flags:  must size, must free, in, by val, */
/* 176 */	NdrFcShort( 0x188 ),	/* x86 Stack size/offset = 392 */
/* 178 */	NdrFcShort( 0x412 ),	/* Type Offset=1042 */

	/* Parameter Var25 */

/* 180 */	NdrFcShort( 0x8b ),	/* Flags:  must size, must free, in, by val, */
/* 182 */	NdrFcShort( 0x198 ),	/* x86 Stack size/offset = 408 */
/* 184 */	NdrFcShort( 0x412 ),	/* Type Offset=1042 */

	/* Parameter Var26 */

/* 186 */	NdrFcShort( 0x8b ),	/* Flags:  must size, must free, in, by val, */
/* 188 */	NdrFcShort( 0x1a8 ),	/* x86 Stack size/offset = 424 */
/* 190 */	NdrFcShort( 0x412 ),	/* Type Offset=1042 */

	/* Parameter Var27 */

/* 192 */	NdrFcShort( 0x8b ),	/* Flags:  must size, must free, in, by val, */
/* 194 */	NdrFcShort( 0x1b8 ),	/* x86 Stack size/offset = 440 */
/* 196 */	NdrFcShort( 0x412 ),	/* Type Offset=1042 */

	/* Parameter Var28 */

/* 198 */	NdrFcShort( 0x8b ),	/* Flags:  must size, must free, in, by val, */
/* 200 */	NdrFcShort( 0x1c8 ),	/* x86 Stack size/offset = 456 */
/* 202 */	NdrFcShort( 0x412 ),	/* Type Offset=1042 */

	/* Parameter Var29 */

/* 204 */	NdrFcShort( 0x8b ),	/* Flags:  must size, must free, in, by val, */
/* 206 */	NdrFcShort( 0x1d8 ),	/* x86 Stack size/offset = 472 */
/* 208 */	NdrFcShort( 0x412 ),	/* Type Offset=1042 */

	/* Parameter Var30 */

/* 210 */	NdrFcShort( 0x8b ),	/* Flags:  must size, must free, in, by val, */
/* 212 */	NdrFcShort( 0x1e8 ),	/* x86 Stack size/offset = 488 */
/* 214 */	NdrFcShort( 0x412 ),	/* Type Offset=1042 */

	/* Parameter Var31 */

/* 216 */	NdrFcShort( 0x8b ),	/* Flags:  must size, must free, in, by val, */
/* 218 */	NdrFcShort( 0x1f8 ),	/* x86 Stack size/offset = 504 */
/* 220 */	NdrFcShort( 0x412 ),	/* Type Offset=1042 */

	/* Return value */

/* 222 */	NdrFcShort( 0x70 ),	/* Flags:  out, return, base type, */
/* 224 */	NdrFcShort( 0x208 ),	/* x86 Stack size/offset = 520 */
/* 226 */	0x8,		/* FC_LONG */
			0x0,		/* 0 */

	/* Procedure MWUnpack */

/* 228 */	0x33,		/* FC_AUTO_HANDLE */
			0x6c,		/* Old Flags:  object, Oi2 */
/* 230 */	NdrFcLong( 0x0 ),	/* 0 */
/* 234 */	NdrFcShort( 0x8 ),	/* 8 */
/* 236 */	NdrFcShort( 0xa0 ),	/* x86 Stack size/offset = 160 */
/* 238 */	NdrFcShort( 0xe ),	/* 14 */
/* 240 */	NdrFcShort( 0x8 ),	/* 8 */
/* 242 */	0x47,		/* Oi2 Flags:  srv must size, clt must size, has return, has ext, */
			0x24,		/* 36 */
/* 244 */	0x8,		/* 8 */
			0x7,		/* Ext Flags:  new corr desc, clt corr check, srv corr check, */
/* 246 */	NdrFcShort( 0x400 ),	/* 1024 */
/* 248 */	NdrFcShort( 0x420 ),	/* 1056 */
/* 250 */	NdrFcShort( 0x0 ),	/* 0 */

	/* Parameter VarArg */

/* 252 */	NdrFcShort( 0x8b ),	/* Flags:  must size, must free, in, by val, */
/* 254 */	NdrFcShort( 0x4 ),	/* x86 Stack size/offset = 4 */
/* 256 */	NdrFcShort( 0x412 ),	/* Type Offset=1042 */

	/* Parameter nStartAt */

/* 258 */	NdrFcShort( 0x48 ),	/* Flags:  in, base type, */
/* 260 */	NdrFcShort( 0x14 ),	/* x86 Stack size/offset = 20 */
/* 262 */	0x8,		/* FC_LONG */
			0x0,		/* 0 */

	/* Parameter bAutoResize */

/* 264 */	NdrFcShort( 0x48 ),	/* Flags:  in, base type, */
/* 266 */	NdrFcShort( 0x18 ),	/* x86 Stack size/offset = 24 */
/* 268 */	0x6,		/* FC_SHORT */
			0x0,		/* 0 */

	/* Parameter pVar0 */

/* 270 */	NdrFcShort( 0x11b ),	/* Flags:  must size, must free, in, out, simple ref, */
/* 272 */	NdrFcShort( 0x1c ),	/* x86 Stack size/offset = 28 */
/* 274 */	NdrFcShort( 0x404 ),	/* Type Offset=1028 */

	/* Parameter pVar1 */

/* 276 */	NdrFcShort( 0x11b ),	/* Flags:  must size, must free, in, out, simple ref, */
/* 278 */	NdrFcShort( 0x20 ),	/* x86 Stack size/offset = 32 */
/* 280 */	NdrFcShort( 0x404 ),	/* Type Offset=1028 */

	/* Parameter pVar2 */

/* 282 */	NdrFcShort( 0x11b ),	/* Flags:  must size, must free, in, out, simple ref, */
/* 284 */	NdrFcShort( 0x24 ),	/* x86 Stack size/offset = 36 */
/* 286 */	NdrFcShort( 0x404 ),	/* Type Offset=1028 */

	/* Parameter pVar3 */

/* 288 */	NdrFcShort( 0x11b ),	/* Flags:  must size, must free, in, out, simple ref, */
/* 290 */	NdrFcShort( 0x28 ),	/* x86 Stack size/offset = 40 */
/* 292 */	NdrFcShort( 0x404 ),	/* Type Offset=1028 */

	/* Parameter pVar4 */

/* 294 */	NdrFcShort( 0x11b ),	/* Flags:  must size, must free, in, out, simple ref, */
/* 296 */	NdrFcShort( 0x2c ),	/* x86 Stack size/offset = 44 */
/* 298 */	NdrFcShort( 0x404 ),	/* Type Offset=1028 */

	/* Parameter pVar5 */

/* 300 */	NdrFcShort( 0x11b ),	/* Flags:  must size, must free, in, out, simple ref, */
/* 302 */	NdrFcShort( 0x30 ),	/* x86 Stack size/offset = 48 */
/* 304 */	NdrFcShort( 0x404 ),	/* Type Offset=1028 */

	/* Parameter pVar6 */

/* 306 */	NdrFcShort( 0x11b ),	/* Flags:  must size, must free, in, out, simple ref, */
/* 308 */	NdrFcShort( 0x34 ),	/* x86 Stack size/offset = 52 */
/* 310 */	NdrFcShort( 0x404 ),	/* Type Offset=1028 */

	/* Parameter pVar7 */

/* 312 */	NdrFcShort( 0x11b ),	/* Flags:  must size, must free, in, out, simple ref, */
/* 314 */	NdrFcShort( 0x38 ),	/* x86 Stack size/offset = 56 */
/* 316 */	NdrFcShort( 0x404 ),	/* Type Offset=1028 */

	/* Parameter pVar8 */

/* 318 */	NdrFcShort( 0x11b ),	/* Flags:  must size, must free, in, out, simple ref, */
/* 320 */	NdrFcShort( 0x3c ),	/* x86 Stack size/offset = 60 */
/* 322 */	NdrFcShort( 0x404 ),	/* Type Offset=1028 */

	/* Parameter pVar9 */

/* 324 */	NdrFcShort( 0x11b ),	/* Flags:  must size, must free, in, out, simple ref, */
/* 326 */	NdrFcShort( 0x40 ),	/* x86 Stack size/offset = 64 */
/* 328 */	NdrFcShort( 0x404 ),	/* Type Offset=1028 */

	/* Parameter pVar10 */

/* 330 */	NdrFcShort( 0x11b ),	/* Flags:  must size, must free, in, out, simple ref, */
/* 332 */	NdrFcShort( 0x44 ),	/* x86 Stack size/offset = 68 */
/* 334 */	NdrFcShort( 0x404 ),	/* Type Offset=1028 */

	/* Parameter pVar11 */

/* 336 */	NdrFcShort( 0x11b ),	/* Flags:  must size, must free, in, out, simple ref, */
/* 338 */	NdrFcShort( 0x48 ),	/* x86 Stack size/offset = 72 */
/* 340 */	NdrFcShort( 0x404 ),	/* Type Offset=1028 */

	/* Parameter pVar12 */

/* 342 */	NdrFcShort( 0x11b ),	/* Flags:  must size, must free, in, out, simple ref, */
/* 344 */	NdrFcShort( 0x4c ),	/* x86 Stack size/offset = 76 */
/* 346 */	NdrFcShort( 0x404 ),	/* Type Offset=1028 */

	/* Parameter pVar13 */

/* 348 */	NdrFcShort( 0x11b ),	/* Flags:  must size, must free, in, out, simple ref, */
/* 350 */	NdrFcShort( 0x50 ),	/* x86 Stack size/offset = 80 */
/* 352 */	NdrFcShort( 0x404 ),	/* Type Offset=1028 */

	/* Parameter pVar14 */

/* 354 */	NdrFcShort( 0x11b ),	/* Flags:  must size, must free, in, out, simple ref, */
/* 356 */	NdrFcShort( 0x54 ),	/* x86 Stack size/offset = 84 */
/* 358 */	NdrFcShort( 0x404 ),	/* Type Offset=1028 */

	/* Parameter pVar15 */

/* 360 */	NdrFcShort( 0x11b ),	/* Flags:  must size, must free, in, out, simple ref, */
/* 362 */	NdrFcShort( 0x58 ),	/* x86 Stack size/offset = 88 */
/* 364 */	NdrFcShort( 0x404 ),	/* Type Offset=1028 */

	/* Parameter pVar16 */

/* 366 */	NdrFcShort( 0x11b ),	/* Flags:  must size, must free, in, out, simple ref, */
/* 368 */	NdrFcShort( 0x5c ),	/* x86 Stack size/offset = 92 */
/* 370 */	NdrFcShort( 0x404 ),	/* Type Offset=1028 */

	/* Parameter pVar17 */

/* 372 */	NdrFcShort( 0x11b ),	/* Flags:  must size, must free, in, out, simple ref, */
/* 374 */	NdrFcShort( 0x60 ),	/* x86 Stack size/offset = 96 */
/* 376 */	NdrFcShort( 0x404 ),	/* Type Offset=1028 */

	/* Parameter pVar18 */

/* 378 */	NdrFcShort( 0x11b ),	/* Flags:  must size, must free, in, out, simple ref, */
/* 380 */	NdrFcShort( 0x64 ),	/* x86 Stack size/offset = 100 */
/* 382 */	NdrFcShort( 0x404 ),	/* Type Offset=1028 */

	/* Parameter pVar19 */

/* 384 */	NdrFcShort( 0x11b ),	/* Flags:  must size, must free, in, out, simple ref, */
/* 386 */	NdrFcShort( 0x68 ),	/* x86 Stack size/offset = 104 */
/* 388 */	NdrFcShort( 0x404 ),	/* Type Offset=1028 */

	/* Parameter pVar20 */

/* 390 */	NdrFcShort( 0x11b ),	/* Flags:  must size, must free, in, out, simple ref, */
/* 392 */	NdrFcShort( 0x6c ),	/* x86 Stack size/offset = 108 */
/* 394 */	NdrFcShort( 0x404 ),	/* Type Offset=1028 */

	/* Parameter pVar21 */

/* 396 */	NdrFcShort( 0x11b ),	/* Flags:  must size, must free, in, out, simple ref, */
/* 398 */	NdrFcShort( 0x70 ),	/* x86 Stack size/offset = 112 */
/* 400 */	NdrFcShort( 0x404 ),	/* Type Offset=1028 */

	/* Parameter pVar22 */

/* 402 */	NdrFcShort( 0x11b ),	/* Flags:  must size, must free, in, out, simple ref, */
/* 404 */	NdrFcShort( 0x74 ),	/* x86 Stack size/offset = 116 */
/* 406 */	NdrFcShort( 0x404 ),	/* Type Offset=1028 */

	/* Parameter pVar23 */

/* 408 */	NdrFcShort( 0x11b ),	/* Flags:  must size, must free, in, out, simple ref, */
/* 410 */	NdrFcShort( 0x78 ),	/* x86 Stack size/offset = 120 */
/* 412 */	NdrFcShort( 0x404 ),	/* Type Offset=1028 */

	/* Parameter pVar24 */

/* 414 */	NdrFcShort( 0x11b ),	/* Flags:  must size, must free, in, out, simple ref, */
/* 416 */	NdrFcShort( 0x7c ),	/* x86 Stack size/offset = 124 */
/* 418 */	NdrFcShort( 0x404 ),	/* Type Offset=1028 */

	/* Parameter pVar25 */

/* 420 */	NdrFcShort( 0x11b ),	/* Flags:  must size, must free, in, out, simple ref, */
/* 422 */	NdrFcShort( 0x80 ),	/* x86 Stack size/offset = 128 */
/* 424 */	NdrFcShort( 0x404 ),	/* Type Offset=1028 */

	/* Parameter pVar26 */

/* 426 */	NdrFcShort( 0x11b ),	/* Flags:  must size, must free, in, out, simple ref, */
/* 428 */	NdrFcShort( 0x84 ),	/* x86 Stack size/offset = 132 */
/* 430 */	NdrFcShort( 0x404 ),	/* Type Offset=1028 */

	/* Parameter pVar27 */

/* 432 */	NdrFcShort( 0x11b ),	/* Flags:  must size, must free, in, out, simple ref, */
/* 434 */	NdrFcShort( 0x88 ),	/* x86 Stack size/offset = 136 */
/* 436 */	NdrFcShort( 0x404 ),	/* Type Offset=1028 */

	/* Parameter pVar28 */

/* 438 */	NdrFcShort( 0x11b ),	/* Flags:  must size, must free, in, out, simple ref, */
/* 440 */	NdrFcShort( 0x8c ),	/* x86 Stack size/offset = 140 */
/* 442 */	NdrFcShort( 0x404 ),	/* Type Offset=1028 */

	/* Parameter pVar29 */

/* 444 */	NdrFcShort( 0x11b ),	/* Flags:  must size, must free, in, out, simple ref, */
/* 446 */	NdrFcShort( 0x90 ),	/* x86 Stack size/offset = 144 */
/* 448 */	NdrFcShort( 0x404 ),	/* Type Offset=1028 */

	/* Parameter pVar30 */

/* 450 */	NdrFcShort( 0x11b ),	/* Flags:  must size, must free, in, out, simple ref, */
/* 452 */	NdrFcShort( 0x94 ),	/* x86 Stack size/offset = 148 */
/* 454 */	NdrFcShort( 0x404 ),	/* Type Offset=1028 */

	/* Parameter pVar31 */

/* 456 */	NdrFcShort( 0x11b ),	/* Flags:  must size, must free, in, out, simple ref, */
/* 458 */	NdrFcShort( 0x98 ),	/* x86 Stack size/offset = 152 */
/* 460 */	NdrFcShort( 0x404 ),	/* Type Offset=1028 */

	/* Return value */

/* 462 */	NdrFcShort( 0x70 ),	/* Flags:  out, return, base type, */
/* 464 */	NdrFcShort( 0x9c ),	/* x86 Stack size/offset = 156 */
/* 466 */	0x8,		/* FC_LONG */
			0x0,		/* 0 */

	/* Procedure MWDate2VariantDate */

/* 468 */	0x33,		/* FC_AUTO_HANDLE */
			0x6c,		/* Old Flags:  object, Oi2 */
/* 470 */	NdrFcLong( 0x0 ),	/* 0 */
/* 474 */	NdrFcShort( 0x9 ),	/* 9 */
/* 476 */	NdrFcShort( 0xc ),	/* x86 Stack size/offset = 12 */
/* 478 */	NdrFcShort( 0x0 ),	/* 0 */
/* 480 */	NdrFcShort( 0x8 ),	/* 8 */
/* 482 */	0x47,		/* Oi2 Flags:  srv must size, clt must size, has return, has ext, */
			0x2,		/* 2 */
/* 484 */	0x8,		/* 8 */
			0x7,		/* Ext Flags:  new corr desc, clt corr check, srv corr check, */
/* 486 */	NdrFcShort( 0x20 ),	/* 32 */
/* 488 */	NdrFcShort( 0x20 ),	/* 32 */
/* 490 */	NdrFcShort( 0x0 ),	/* 0 */

	/* Parameter pVar */

/* 492 */	NdrFcShort( 0x11b ),	/* Flags:  must size, must free, in, out, simple ref, */
/* 494 */	NdrFcShort( 0x4 ),	/* x86 Stack size/offset = 4 */
/* 496 */	NdrFcShort( 0x404 ),	/* Type Offset=1028 */

	/* Return value */

/* 498 */	NdrFcShort( 0x70 ),	/* Flags:  out, return, base type, */
/* 500 */	NdrFcShort( 0x8 ),	/* x86 Stack size/offset = 8 */
/* 502 */	0x8,		/* FC_LONG */
			0x0,		/* 0 */

	/* Procedure MWInitApplication */

/* 504 */	0x33,		/* FC_AUTO_HANDLE */
			0x6c,		/* Old Flags:  object, Oi2 */
/* 506 */	NdrFcLong( 0x0 ),	/* 0 */
/* 510 */	NdrFcShort( 0xa ),	/* 10 */
/* 512 */	NdrFcShort( 0xc ),	/* x86 Stack size/offset = 12 */
/* 514 */	NdrFcShort( 0x0 ),	/* 0 */
/* 516 */	NdrFcShort( 0x8 ),	/* 8 */
/* 518 */	0x46,		/* Oi2 Flags:  clt must size, has return, has ext, */
			0x2,		/* 2 */
/* 520 */	0x8,		/* 8 */
			0x1,		/* Ext Flags:  new corr desc, */
/* 522 */	NdrFcShort( 0x0 ),	/* 0 */
/* 524 */	NdrFcShort( 0x0 ),	/* 0 */
/* 526 */	NdrFcShort( 0x0 ),	/* 0 */

	/* Parameter pApp */

/* 528 */	NdrFcShort( 0xb ),	/* Flags:  must size, must free, in, */
/* 530 */	NdrFcShort( 0x4 ),	/* x86 Stack size/offset = 4 */
/* 532 */	NdrFcShort( 0x166 ),	/* Type Offset=358 */

	/* Return value */

/* 534 */	NdrFcShort( 0x70 ),	/* Flags:  out, return, base type, */
/* 536 */	NdrFcShort( 0x8 ),	/* x86 Stack size/offset = 8 */
/* 538 */	0x8,		/* FC_LONG */
			0x0,		/* 0 */

			0x0
        }
    };

static const MIDL_TYPE_FORMAT_STRING __MIDL_TypeFormatString =
    {
        0,
        {
			NdrFcShort( 0x0 ),	/* 0 */
/*  2 */	
			0x11, 0x4,	/* FC_RP [alloced_on_stack] */
/*  4 */	NdrFcShort( 0x400 ),	/* Offset= 1024 (1028) */
/*  6 */	
			0x13, 0x0,	/* FC_OP */
/*  8 */	NdrFcShort( 0x3e8 ),	/* Offset= 1000 (1008) */
/* 10 */	
			0x2b,		/* FC_NON_ENCAPSULATED_UNION */
			0x9,		/* FC_ULONG */
/* 12 */	0x7,		/* Corr desc: FC_USHORT */
			0x0,		/*  */
/* 14 */	NdrFcShort( 0xfff8 ),	/* -8 */
/* 16 */	NdrFcShort( 0x1 ),	/* Corr flags:  early, */
/* 18 */	NdrFcShort( 0x2 ),	/* Offset= 2 (20) */
/* 20 */	NdrFcShort( 0x10 ),	/* 16 */
/* 22 */	NdrFcShort( 0x2f ),	/* 47 */
/* 24 */	NdrFcLong( 0x14 ),	/* 20 */
/* 28 */	NdrFcShort( 0x800b ),	/* Simple arm type: FC_HYPER */
/* 30 */	NdrFcLong( 0x3 ),	/* 3 */
/* 34 */	NdrFcShort( 0x8008 ),	/* Simple arm type: FC_LONG */
/* 36 */	NdrFcLong( 0x11 ),	/* 17 */
/* 40 */	NdrFcShort( 0x8001 ),	/* Simple arm type: FC_BYTE */
/* 42 */	NdrFcLong( 0x2 ),	/* 2 */
/* 46 */	NdrFcShort( 0x8006 ),	/* Simple arm type: FC_SHORT */
/* 48 */	NdrFcLong( 0x4 ),	/* 4 */
/* 52 */	NdrFcShort( 0x800a ),	/* Simple arm type: FC_FLOAT */
/* 54 */	NdrFcLong( 0x5 ),	/* 5 */
/* 58 */	NdrFcShort( 0x800c ),	/* Simple arm type: FC_DOUBLE */
/* 60 */	NdrFcLong( 0xb ),	/* 11 */
/* 64 */	NdrFcShort( 0x8006 ),	/* Simple arm type: FC_SHORT */
/* 66 */	NdrFcLong( 0xa ),	/* 10 */
/* 70 */	NdrFcShort( 0x8008 ),	/* Simple arm type: FC_LONG */
/* 72 */	NdrFcLong( 0x6 ),	/* 6 */
/* 76 */	NdrFcShort( 0xe8 ),	/* Offset= 232 (308) */
/* 78 */	NdrFcLong( 0x7 ),	/* 7 */
/* 82 */	NdrFcShort( 0x800c ),	/* Simple arm type: FC_DOUBLE */
/* 84 */	NdrFcLong( 0x8 ),	/* 8 */
/* 88 */	NdrFcShort( 0xe2 ),	/* Offset= 226 (314) */
/* 90 */	NdrFcLong( 0xd ),	/* 13 */
/* 94 */	NdrFcShort( 0xf6 ),	/* Offset= 246 (340) */
/* 96 */	NdrFcLong( 0x9 ),	/* 9 */
/* 100 */	NdrFcShort( 0x102 ),	/* Offset= 258 (358) */
/* 102 */	NdrFcLong( 0x2000 ),	/* 8192 */
/* 106 */	NdrFcShort( 0x10e ),	/* Offset= 270 (376) */
/* 108 */	NdrFcLong( 0x24 ),	/* 36 */
/* 112 */	NdrFcShort( 0x336 ),	/* Offset= 822 (934) */
/* 114 */	NdrFcLong( 0x4024 ),	/* 16420 */
/* 118 */	NdrFcShort( 0x330 ),	/* Offset= 816 (934) */
/* 120 */	NdrFcLong( 0x4011 ),	/* 16401 */
/* 124 */	NdrFcShort( 0x32e ),	/* Offset= 814 (938) */
/* 126 */	NdrFcLong( 0x4002 ),	/* 16386 */
/* 130 */	NdrFcShort( 0x32c ),	/* Offset= 812 (942) */
/* 132 */	NdrFcLong( 0x4003 ),	/* 16387 */
/* 136 */	NdrFcShort( 0x32a ),	/* Offset= 810 (946) */
/* 138 */	NdrFcLong( 0x4014 ),	/* 16404 */
/* 142 */	NdrFcShort( 0x328 ),	/* Offset= 808 (950) */
/* 144 */	NdrFcLong( 0x4004 ),	/* 16388 */
/* 148 */	NdrFcShort( 0x326 ),	/* Offset= 806 (954) */
/* 150 */	NdrFcLong( 0x4005 ),	/* 16389 */
/* 154 */	NdrFcShort( 0x324 ),	/* Offset= 804 (958) */
/* 156 */	NdrFcLong( 0x400b ),	/* 16395 */
/* 160 */	NdrFcShort( 0x30e ),	/* Offset= 782 (942) */
/* 162 */	NdrFcLong( 0x400a ),	/* 16394 */
/* 166 */	NdrFcShort( 0x30c ),	/* Offset= 780 (946) */
/* 168 */	NdrFcLong( 0x4006 ),	/* 16390 */
/* 172 */	NdrFcShort( 0x316 ),	/* Offset= 790 (962) */
/* 174 */	NdrFcLong( 0x4007 ),	/* 16391 */
/* 178 */	NdrFcShort( 0x30c ),	/* Offset= 780 (958) */
/* 180 */	NdrFcLong( 0x4008 ),	/* 16392 */
/* 184 */	NdrFcShort( 0x30e ),	/* Offset= 782 (966) */
/* 186 */	NdrFcLong( 0x400d ),	/* 16397 */
/* 190 */	NdrFcShort( 0x30c ),	/* Offset= 780 (970) */
/* 192 */	NdrFcLong( 0x4009 ),	/* 16393 */
/* 196 */	NdrFcShort( 0x30a ),	/* Offset= 778 (974) */
/* 198 */	NdrFcLong( 0x6000 ),	/* 24576 */
/* 202 */	NdrFcShort( 0x308 ),	/* Offset= 776 (978) */
/* 204 */	NdrFcLong( 0x400c ),	/* 16396 */
/* 208 */	NdrFcShort( 0x306 ),	/* Offset= 774 (982) */
/* 210 */	NdrFcLong( 0x10 ),	/* 16 */
/* 214 */	NdrFcShort( 0x8002 ),	/* Simple arm type: FC_CHAR */
/* 216 */	NdrFcLong( 0x12 ),	/* 18 */
/* 220 */	NdrFcShort( 0x8006 ),	/* Simple arm type: FC_SHORT */
/* 222 */	NdrFcLong( 0x13 ),	/* 19 */
/* 226 */	NdrFcShort( 0x8008 ),	/* Simple arm type: FC_LONG */
/* 228 */	NdrFcLong( 0x15 ),	/* 21 */
/* 232 */	NdrFcShort( 0x800b ),	/* Simple arm type: FC_HYPER */
/* 234 */	NdrFcLong( 0x16 ),	/* 22 */
/* 238 */	NdrFcShort( 0x8008 ),	/* Simple arm type: FC_LONG */
/* 240 */	NdrFcLong( 0x17 ),	/* 23 */
/* 244 */	NdrFcShort( 0x8008 ),	/* Simple arm type: FC_LONG */
/* 246 */	NdrFcLong( 0xe ),	/* 14 */
/* 250 */	NdrFcShort( 0x2e4 ),	/* Offset= 740 (990) */
/* 252 */	NdrFcLong( 0x400e ),	/* 16398 */
/* 256 */	NdrFcShort( 0x2e8 ),	/* Offset= 744 (1000) */
/* 258 */	NdrFcLong( 0x4010 ),	/* 16400 */
/* 262 */	NdrFcShort( 0x2e6 ),	/* Offset= 742 (1004) */
/* 264 */	NdrFcLong( 0x4012 ),	/* 16402 */
/* 268 */	NdrFcShort( 0x2a2 ),	/* Offset= 674 (942) */
/* 270 */	NdrFcLong( 0x4013 ),	/* 16403 */
/* 274 */	NdrFcShort( 0x2a0 ),	/* Offset= 672 (946) */
/* 276 */	NdrFcLong( 0x4015 ),	/* 16405 */
/* 280 */	NdrFcShort( 0x29e ),	/* Offset= 670 (950) */
/* 282 */	NdrFcLong( 0x4016 ),	/* 16406 */
/* 286 */	NdrFcShort( 0x294 ),	/* Offset= 660 (946) */
/* 288 */	NdrFcLong( 0x4017 ),	/* 16407 */
/* 292 */	NdrFcShort( 0x28e ),	/* Offset= 654 (946) */
/* 294 */	NdrFcLong( 0x0 ),	/* 0 */
/* 298 */	NdrFcShort( 0x0 ),	/* Offset= 0 (298) */
/* 300 */	NdrFcLong( 0x1 ),	/* 1 */
/* 304 */	NdrFcShort( 0x0 ),	/* Offset= 0 (304) */
/* 306 */	NdrFcShort( 0xffff ),	/* Offset= -1 (305) */
/* 308 */	
			0x15,		/* FC_STRUCT */
			0x7,		/* 7 */
/* 310 */	NdrFcShort( 0x8 ),	/* 8 */
/* 312 */	0xb,		/* FC_HYPER */
			0x5b,		/* FC_END */
/* 314 */	
			0x13, 0x0,	/* FC_OP */
/* 316 */	NdrFcShort( 0xe ),	/* Offset= 14 (330) */
/* 318 */	
			0x1b,		/* FC_CARRAY */
			0x1,		/* 1 */
/* 320 */	NdrFcShort( 0x2 ),	/* 2 */
/* 322 */	0x9,		/* Corr desc: FC_ULONG */
			0x0,		/*  */
/* 324 */	NdrFcShort( 0xfffc ),	/* -4 */
/* 326 */	NdrFcShort( 0x1 ),	/* Corr flags:  early, */
/* 328 */	0x6,		/* FC_SHORT */
			0x5b,		/* FC_END */
/* 330 */	
			0x17,		/* FC_CSTRUCT */
			0x3,		/* 3 */
/* 332 */	NdrFcShort( 0x8 ),	/* 8 */
/* 334 */	NdrFcShort( 0xfff0 ),	/* Offset= -16 (318) */
/* 336 */	0x8,		/* FC_LONG */
			0x8,		/* FC_LONG */
/* 338 */	0x5c,		/* FC_PAD */
			0x5b,		/* FC_END */
/* 340 */	
			0x2f,		/* FC_IP */
			0x5a,		/* FC_CONSTANT_IID */
/* 342 */	NdrFcLong( 0x0 ),	/* 0 */
/* 346 */	NdrFcShort( 0x0 ),	/* 0 */
/* 348 */	NdrFcShort( 0x0 ),	/* 0 */
/* 350 */	0xc0,		/* 192 */
			0x0,		/* 0 */
/* 352 */	0x0,		/* 0 */
			0x0,		/* 0 */
/* 354 */	0x0,		/* 0 */
			0x0,		/* 0 */
/* 356 */	0x0,		/* 0 */
			0x46,		/* 70 */
/* 358 */	
			0x2f,		/* FC_IP */
			0x5a,		/* FC_CONSTANT_IID */
/* 360 */	NdrFcLong( 0x20400 ),	/* 132096 */
/* 364 */	NdrFcShort( 0x0 ),	/* 0 */
/* 366 */	NdrFcShort( 0x0 ),	/* 0 */
/* 368 */	0xc0,		/* 192 */
			0x0,		/* 0 */
/* 370 */	0x0,		/* 0 */
			0x0,		/* 0 */
/* 372 */	0x0,		/* 0 */
			0x0,		/* 0 */
/* 374 */	0x0,		/* 0 */
			0x46,		/* 70 */
/* 376 */	
			0x13, 0x10,	/* FC_OP [pointer_deref] */
/* 378 */	NdrFcShort( 0x2 ),	/* Offset= 2 (380) */
/* 380 */	
			0x13, 0x0,	/* FC_OP */
/* 382 */	NdrFcShort( 0x216 ),	/* Offset= 534 (916) */
/* 384 */	
			0x2a,		/* FC_ENCAPSULATED_UNION */
			0x49,		/* 73 */
/* 386 */	NdrFcShort( 0x18 ),	/* 24 */
/* 388 */	NdrFcShort( 0xa ),	/* 10 */
/* 390 */	NdrFcLong( 0x8 ),	/* 8 */
/* 394 */	NdrFcShort( 0x5a ),	/* Offset= 90 (484) */
/* 396 */	NdrFcLong( 0xd ),	/* 13 */
/* 400 */	NdrFcShort( 0x7e ),	/* Offset= 126 (526) */
/* 402 */	NdrFcLong( 0x9 ),	/* 9 */
/* 406 */	NdrFcShort( 0x9e ),	/* Offset= 158 (564) */
/* 408 */	NdrFcLong( 0xc ),	/* 12 */
/* 412 */	NdrFcShort( 0xc8 ),	/* Offset= 200 (612) */
/* 414 */	NdrFcLong( 0x24 ),	/* 36 */
/* 418 */	NdrFcShort( 0x124 ),	/* Offset= 292 (710) */
/* 420 */	NdrFcLong( 0x800d ),	/* 32781 */
/* 424 */	NdrFcShort( 0x140 ),	/* Offset= 320 (744) */
/* 426 */	NdrFcLong( 0x10 ),	/* 16 */
/* 430 */	NdrFcShort( 0x15a ),	/* Offset= 346 (776) */
/* 432 */	NdrFcLong( 0x2 ),	/* 2 */
/* 436 */	NdrFcShort( 0x174 ),	/* Offset= 372 (808) */
/* 438 */	NdrFcLong( 0x3 ),	/* 3 */
/* 442 */	NdrFcShort( 0x18e ),	/* Offset= 398 (840) */
/* 444 */	NdrFcLong( 0x14 ),	/* 20 */
/* 448 */	NdrFcShort( 0x1a8 ),	/* Offset= 424 (872) */
/* 450 */	NdrFcShort( 0xffff ),	/* Offset= -1 (449) */
/* 452 */	
			0x1b,		/* FC_CARRAY */
			0x3,		/* 3 */
/* 454 */	NdrFcShort( 0x4 ),	/* 4 */
/* 456 */	0x19,		/* Corr desc:  field pointer, FC_ULONG */
			0x0,		/*  */
/* 458 */	NdrFcShort( 0x0 ),	/* 0 */
/* 460 */	NdrFcShort( 0x1 ),	/* Corr flags:  early, */
/* 462 */	
			0x4b,		/* FC_PP */
			0x5c,		/* FC_PAD */
/* 464 */	
			0x48,		/* FC_VARIABLE_REPEAT */
			0x49,		/* FC_FIXED_OFFSET */
/* 466 */	NdrFcShort( 0x4 ),	/* 4 */
/* 468 */	NdrFcShort( 0x0 ),	/* 0 */
/* 470 */	NdrFcShort( 0x1 ),	/* 1 */
/* 472 */	NdrFcShort( 0x0 ),	/* 0 */
/* 474 */	NdrFcShort( 0x0 ),	/* 0 */
/* 476 */	0x13, 0x0,	/* FC_OP */
/* 478 */	NdrFcShort( 0xff6c ),	/* Offset= -148 (330) */
/* 480 */	
			0x5b,		/* FC_END */

			0x8,		/* FC_LONG */
/* 482 */	0x5c,		/* FC_PAD */
			0x5b,		/* FC_END */
/* 484 */	
			0x16,		/* FC_PSTRUCT */
			0x3,		/* 3 */
/* 486 */	NdrFcShort( 0x8 ),	/* 8 */
/* 488 */	
			0x4b,		/* FC_PP */
			0x5c,		/* FC_PAD */
/* 490 */	
			0x46,		/* FC_NO_REPEAT */
			0x5c,		/* FC_PAD */
/* 492 */	NdrFcShort( 0x4 ),	/* 4 */
/* 494 */	NdrFcShort( 0x4 ),	/* 4 */
/* 496 */	0x11, 0x0,	/* FC_RP */
/* 498 */	NdrFcShort( 0xffd2 ),	/* Offset= -46 (452) */
/* 500 */	
			0x5b,		/* FC_END */

			0x8,		/* FC_LONG */
/* 502 */	0x8,		/* FC_LONG */
			0x5b,		/* FC_END */
/* 504 */	
			0x21,		/* FC_BOGUS_ARRAY */
			0x3,		/* 3 */
/* 506 */	NdrFcShort( 0x0 ),	/* 0 */
/* 508 */	0x19,		/* Corr desc:  field pointer, FC_ULONG */
			0x0,		/*  */
/* 510 */	NdrFcShort( 0x0 ),	/* 0 */
/* 512 */	NdrFcShort( 0x1 ),	/* Corr flags:  early, */
/* 514 */	NdrFcLong( 0xffffffff ),	/* -1 */
/* 518 */	NdrFcShort( 0x0 ),	/* Corr flags:  */
/* 520 */	0x4c,		/* FC_EMBEDDED_COMPLEX */
			0x0,		/* 0 */
/* 522 */	NdrFcShort( 0xff4a ),	/* Offset= -182 (340) */
/* 524 */	0x5c,		/* FC_PAD */
			0x5b,		/* FC_END */
/* 526 */	
			0x1a,		/* FC_BOGUS_STRUCT */
			0x3,		/* 3 */
/* 528 */	NdrFcShort( 0x8 ),	/* 8 */
/* 530 */	NdrFcShort( 0x0 ),	/* 0 */
/* 532 */	NdrFcShort( 0x6 ),	/* Offset= 6 (538) */
/* 534 */	0x8,		/* FC_LONG */
			0x36,		/* FC_POINTER */
/* 536 */	0x5c,		/* FC_PAD */
			0x5b,		/* FC_END */
/* 538 */	
			0x11, 0x0,	/* FC_RP */
/* 540 */	NdrFcShort( 0xffdc ),	/* Offset= -36 (504) */
/* 542 */	
			0x21,		/* FC_BOGUS_ARRAY */
			0x3,		/* 3 */
/* 544 */	NdrFcShort( 0x0 ),	/* 0 */
/* 546 */	0x19,		/* Corr desc:  field pointer, FC_ULONG */
			0x0,		/*  */
/* 548 */	NdrFcShort( 0x0 ),	/* 0 */
/* 550 */	NdrFcShort( 0x1 ),	/* Corr flags:  early, */
/* 552 */	NdrFcLong( 0xffffffff ),	/* -1 */
/* 556 */	NdrFcShort( 0x0 ),	/* Corr flags:  */
/* 558 */	0x4c,		/* FC_EMBEDDED_COMPLEX */
			0x0,		/* 0 */
/* 560 */	NdrFcShort( 0xff36 ),	/* Offset= -202 (358) */
/* 562 */	0x5c,		/* FC_PAD */
			0x5b,		/* FC_END */
/* 564 */	
			0x1a,		/* FC_BOGUS_STRUCT */
			0x3,		/* 3 */
/* 566 */	NdrFcShort( 0x8 ),	/* 8 */
/* 568 */	NdrFcShort( 0x0 ),	/* 0 */
/* 570 */	NdrFcShort( 0x6 ),	/* Offset= 6 (576) */
/* 572 */	0x8,		/* FC_LONG */
			0x36,		/* FC_POINTER */
/* 574 */	0x5c,		/* FC_PAD */
			0x5b,		/* FC_END */
/* 576 */	
			0x11, 0x0,	/* FC_RP */
/* 578 */	NdrFcShort( 0xffdc ),	/* Offset= -36 (542) */
/* 580 */	
			0x1b,		/* FC_CARRAY */
			0x3,		/* 3 */
/* 582 */	NdrFcShort( 0x4 ),	/* 4 */
/* 584 */	0x19,		/* Corr desc:  field pointer, FC_ULONG */
			0x0,		/*  */
/* 586 */	NdrFcShort( 0x0 ),	/* 0 */
/* 588 */	NdrFcShort( 0x1 ),	/* Corr flags:  early, */
/* 590 */	
			0x4b,		/* FC_PP */
			0x5c,		/* FC_PAD */
/* 592 */	
			0x48,		/* FC_VARIABLE_REPEAT */
			0x49,		/* FC_FIXED_OFFSET */
/* 594 */	NdrFcShort( 0x4 ),	/* 4 */
/* 596 */	NdrFcShort( 0x0 ),	/* 0 */
/* 598 */	NdrFcShort( 0x1 ),	/* 1 */
/* 600 */	NdrFcShort( 0x0 ),	/* 0 */
/* 602 */	NdrFcShort( 0x0 ),	/* 0 */
/* 604 */	0x13, 0x0,	/* FC_OP */
/* 606 */	NdrFcShort( 0x192 ),	/* Offset= 402 (1008) */
/* 608 */	
			0x5b,		/* FC_END */

			0x8,		/* FC_LONG */
/* 610 */	0x5c,		/* FC_PAD */
			0x5b,		/* FC_END */
/* 612 */	
			0x1a,		/* FC_BOGUS_STRUCT */
			0x3,		/* 3 */
/* 614 */	NdrFcShort( 0x8 ),	/* 8 */
/* 616 */	NdrFcShort( 0x0 ),	/* 0 */
/* 618 */	NdrFcShort( 0x6 ),	/* Offset= 6 (624) */
/* 620 */	0x8,		/* FC_LONG */
			0x36,		/* FC_POINTER */
/* 622 */	0x5c,		/* FC_PAD */
			0x5b,		/* FC_END */
/* 624 */	
			0x11, 0x0,	/* FC_RP */
/* 626 */	NdrFcShort( 0xffd2 ),	/* Offset= -46 (580) */
/* 628 */	
			0x2f,		/* FC_IP */
			0x5a,		/* FC_CONSTANT_IID */
/* 630 */	NdrFcLong( 0x2f ),	/* 47 */
/* 634 */	NdrFcShort( 0x0 ),	/* 0 */
/* 636 */	NdrFcShort( 0x0 ),	/* 0 */
/* 638 */	0xc0,		/* 192 */
			0x0,		/* 0 */
/* 640 */	0x0,		/* 0 */
			0x0,		/* 0 */
/* 642 */	0x0,		/* 0 */
			0x0,		/* 0 */
/* 644 */	0x0,		/* 0 */
			0x46,		/* 70 */
/* 646 */	
			0x1b,		/* FC_CARRAY */
			0x0,		/* 0 */
/* 648 */	NdrFcShort( 0x1 ),	/* 1 */
/* 650 */	0x19,		/* Corr desc:  field pointer, FC_ULONG */
			0x0,		/*  */
/* 652 */	NdrFcShort( 0x4 ),	/* 4 */
/* 654 */	NdrFcShort( 0x1 ),	/* Corr flags:  early, */
/* 656 */	0x1,		/* FC_BYTE */
			0x5b,		/* FC_END */
/* 658 */	
			0x1a,		/* FC_BOGUS_STRUCT */
			0x3,		/* 3 */
/* 660 */	NdrFcShort( 0x10 ),	/* 16 */
/* 662 */	NdrFcShort( 0x0 ),	/* 0 */
/* 664 */	NdrFcShort( 0xa ),	/* Offset= 10 (674) */
/* 666 */	0x8,		/* FC_LONG */
			0x8,		/* FC_LONG */
/* 668 */	0x4c,		/* FC_EMBEDDED_COMPLEX */
			0x0,		/* 0 */
/* 670 */	NdrFcShort( 0xffd6 ),	/* Offset= -42 (628) */
/* 672 */	0x36,		/* FC_POINTER */
			0x5b,		/* FC_END */
/* 674 */	
			0x13, 0x0,	/* FC_OP */
/* 676 */	NdrFcShort( 0xffe2 ),	/* Offset= -30 (646) */
/* 678 */	
			0x1b,		/* FC_CARRAY */
			0x3,		/* 3 */
/* 680 */	NdrFcShort( 0x4 ),	/* 4 */
/* 682 */	0x19,		/* Corr desc:  field pointer, FC_ULONG */
			0x0,		/*  */
/* 684 */	NdrFcShort( 0x0 ),	/* 0 */
/* 686 */	NdrFcShort( 0x1 ),	/* Corr flags:  early, */
/* 688 */	
			0x4b,		/* FC_PP */
			0x5c,		/* FC_PAD */
/* 690 */	
			0x48,		/* FC_VARIABLE_REPEAT */
			0x49,		/* FC_FIXED_OFFSET */
/* 692 */	NdrFcShort( 0x4 ),	/* 4 */
/* 694 */	NdrFcShort( 0x0 ),	/* 0 */
/* 696 */	NdrFcShort( 0x1 ),	/* 1 */
/* 698 */	NdrFcShort( 0x0 ),	/* 0 */
/* 700 */	NdrFcShort( 0x0 ),	/* 0 */
/* 702 */	0x13, 0x0,	/* FC_OP */
/* 704 */	NdrFcShort( 0xffd2 ),	/* Offset= -46 (658) */
/* 706 */	
			0x5b,		/* FC_END */

			0x8,		/* FC_LONG */
/* 708 */	0x5c,		/* FC_PAD */
			0x5b,		/* FC_END */
/* 710 */	
			0x1a,		/* FC_BOGUS_STRUCT */
			0x3,		/* 3 */
/* 712 */	NdrFcShort( 0x8 ),	/* 8 */
/* 714 */	NdrFcShort( 0x0 ),	/* 0 */
/* 716 */	NdrFcShort( 0x6 ),	/* Offset= 6 (722) */
/* 718 */	0x8,		/* FC_LONG */
			0x36,		/* FC_POINTER */
/* 720 */	0x5c,		/* FC_PAD */
			0x5b,		/* FC_END */
/* 722 */	
			0x11, 0x0,	/* FC_RP */
/* 724 */	NdrFcShort( 0xffd2 ),	/* Offset= -46 (678) */
/* 726 */	
			0x1d,		/* FC_SMFARRAY */
			0x0,		/* 0 */
/* 728 */	NdrFcShort( 0x8 ),	/* 8 */
/* 730 */	0x1,		/* FC_BYTE */
			0x5b,		/* FC_END */
/* 732 */	
			0x15,		/* FC_STRUCT */
			0x3,		/* 3 */
/* 734 */	NdrFcShort( 0x10 ),	/* 16 */
/* 736 */	0x8,		/* FC_LONG */
			0x6,		/* FC_SHORT */
/* 738 */	0x6,		/* FC_SHORT */
			0x4c,		/* FC_EMBEDDED_COMPLEX */
/* 740 */	0x0,		/* 0 */
			NdrFcShort( 0xfff1 ),	/* Offset= -15 (726) */
			0x5b,		/* FC_END */
/* 744 */	
			0x1a,		/* FC_BOGUS_STRUCT */
			0x3,		/* 3 */
/* 746 */	NdrFcShort( 0x18 ),	/* 24 */
/* 748 */	NdrFcShort( 0x0 ),	/* 0 */
/* 750 */	NdrFcShort( 0xa ),	/* Offset= 10 (760) */
/* 752 */	0x8,		/* FC_LONG */
			0x36,		/* FC_POINTER */
/* 754 */	0x4c,		/* FC_EMBEDDED_COMPLEX */
			0x0,		/* 0 */
/* 756 */	NdrFcShort( 0xffe8 ),	/* Offset= -24 (732) */
/* 758 */	0x5c,		/* FC_PAD */
			0x5b,		/* FC_END */
/* 760 */	
			0x11, 0x0,	/* FC_RP */
/* 762 */	NdrFcShort( 0xfefe ),	/* Offset= -258 (504) */
/* 764 */	
			0x1b,		/* FC_CARRAY */
			0x0,		/* 0 */
/* 766 */	NdrFcShort( 0x1 ),	/* 1 */
/* 768 */	0x19,		/* Corr desc:  field pointer, FC_ULONG */
			0x0,		/*  */
/* 770 */	NdrFcShort( 0x0 ),	/* 0 */
/* 772 */	NdrFcShort( 0x1 ),	/* Corr flags:  early, */
/* 774 */	0x1,		/* FC_BYTE */
			0x5b,		/* FC_END */
/* 776 */	
			0x16,		/* FC_PSTRUCT */
			0x3,		/* 3 */
/* 778 */	NdrFcShort( 0x8 ),	/* 8 */
/* 780 */	
			0x4b,		/* FC_PP */
			0x5c,		/* FC_PAD */
/* 782 */	
			0x46,		/* FC_NO_REPEAT */
			0x5c,		/* FC_PAD */
/* 784 */	NdrFcShort( 0x4 ),	/* 4 */
/* 786 */	NdrFcShort( 0x4 ),	/* 4 */
/* 788 */	0x13, 0x0,	/* FC_OP */
/* 790 */	NdrFcShort( 0xffe6 ),	/* Offset= -26 (764) */
/* 792 */	
			0x5b,		/* FC_END */

			0x8,		/* FC_LONG */
/* 794 */	0x8,		/* FC_LONG */
			0x5b,		/* FC_END */
/* 796 */	
			0x1b,		/* FC_CARRAY */
			0x1,		/* 1 */
/* 798 */	NdrFcShort( 0x2 ),	/* 2 */
/* 800 */	0x19,		/* Corr desc:  field pointer, FC_ULONG */
			0x0,		/*  */
/* 802 */	NdrFcShort( 0x0 ),	/* 0 */
/* 804 */	NdrFcShort( 0x1 ),	/* Corr flags:  early, */
/* 806 */	0x6,		/* FC_SHORT */
			0x5b,		/* FC_END */
/* 808 */	
			0x16,		/* FC_PSTRUCT */
			0x3,		/* 3 */
/* 810 */	NdrFcShort( 0x8 ),	/* 8 */
/* 812 */	
			0x4b,		/* FC_PP */
			0x5c,		/* FC_PAD */
/* 814 */	
			0x46,		/* FC_NO_REPEAT */
			0x5c,		/* FC_PAD */
/* 816 */	NdrFcShort( 0x4 ),	/* 4 */
/* 818 */	NdrFcShort( 0x4 ),	/* 4 */
/* 820 */	0x13, 0x0,	/* FC_OP */
/* 822 */	NdrFcShort( 0xffe6 ),	/* Offset= -26 (796) */
/* 824 */	
			0x5b,		/* FC_END */

			0x8,		/* FC_LONG */
/* 826 */	0x8,		/* FC_LONG */
			0x5b,		/* FC_END */
/* 828 */	
			0x1b,		/* FC_CARRAY */
			0x3,		/* 3 */
/* 830 */	NdrFcShort( 0x4 ),	/* 4 */
/* 832 */	0x19,		/* Corr desc:  field pointer, FC_ULONG */
			0x0,		/*  */
/* 834 */	NdrFcShort( 0x0 ),	/* 0 */
/* 836 */	NdrFcShort( 0x1 ),	/* Corr flags:  early, */
/* 838 */	0x8,		/* FC_LONG */
			0x5b,		/* FC_END */
/* 840 */	
			0x16,		/* FC_PSTRUCT */
			0x3,		/* 3 */
/* 842 */	NdrFcShort( 0x8 ),	/* 8 */
/* 844 */	
			0x4b,		/* FC_PP */
			0x5c,		/* FC_PAD */
/* 846 */	
			0x46,		/* FC_NO_REPEAT */
			0x5c,		/* FC_PAD */
/* 848 */	NdrFcShort( 0x4 ),	/* 4 */
/* 850 */	NdrFcShort( 0x4 ),	/* 4 */
/* 852 */	0x13, 0x0,	/* FC_OP */
/* 854 */	NdrFcShort( 0xffe6 ),	/* Offset= -26 (828) */
/* 856 */	
			0x5b,		/* FC_END */

			0x8,		/* FC_LONG */
/* 858 */	0x8,		/* FC_LONG */
			0x5b,		/* FC_END */
/* 860 */	
			0x1b,		/* FC_CARRAY */
			0x7,		/* 7 */
/* 862 */	NdrFcShort( 0x8 ),	/* 8 */
/* 864 */	0x19,		/* Corr desc:  field pointer, FC_ULONG */
			0x0,		/*  */
/* 866 */	NdrFcShort( 0x0 ),	/* 0 */
/* 868 */	NdrFcShort( 0x1 ),	/* Corr flags:  early, */
/* 870 */	0xb,		/* FC_HYPER */
			0x5b,		/* FC_END */
/* 872 */	
			0x16,		/* FC_PSTRUCT */
			0x3,		/* 3 */
/* 874 */	NdrFcShort( 0x8 ),	/* 8 */
/* 876 */	
			0x4b,		/* FC_PP */
			0x5c,		/* FC_PAD */
/* 878 */	
			0x46,		/* FC_NO_REPEAT */
			0x5c,		/* FC_PAD */
/* 880 */	NdrFcShort( 0x4 ),	/* 4 */
/* 882 */	NdrFcShort( 0x4 ),	/* 4 */
/* 884 */	0x13, 0x0,	/* FC_OP */
/* 886 */	NdrFcShort( 0xffe6 ),	/* Offset= -26 (860) */
/* 888 */	
			0x5b,		/* FC_END */

			0x8,		/* FC_LONG */
/* 890 */	0x8,		/* FC_LONG */
			0x5b,		/* FC_END */
/* 892 */	
			0x15,		/* FC_STRUCT */
			0x3,		/* 3 */
/* 894 */	NdrFcShort( 0x8 ),	/* 8 */
/* 896 */	0x8,		/* FC_LONG */
			0x8,		/* FC_LONG */
/* 898 */	0x5c,		/* FC_PAD */
			0x5b,		/* FC_END */
/* 900 */	
			0x1b,		/* FC_CARRAY */
			0x3,		/* 3 */
/* 902 */	NdrFcShort( 0x8 ),	/* 8 */
/* 904 */	0x7,		/* Corr desc: FC_USHORT */
			0x0,		/*  */
/* 906 */	NdrFcShort( 0xffd8 ),	/* -40 */
/* 908 */	NdrFcShort( 0x1 ),	/* Corr flags:  early, */
/* 910 */	0x4c,		/* FC_EMBEDDED_COMPLEX */
			0x0,		/* 0 */
/* 912 */	NdrFcShort( 0xffec ),	/* Offset= -20 (892) */
/* 914 */	0x5c,		/* FC_PAD */
			0x5b,		/* FC_END */
/* 916 */	
			0x1a,		/* FC_BOGUS_STRUCT */
			0x3,		/* 3 */
/* 918 */	NdrFcShort( 0x28 ),	/* 40 */
/* 920 */	NdrFcShort( 0xffec ),	/* Offset= -20 (900) */
/* 922 */	NdrFcShort( 0x0 ),	/* Offset= 0 (922) */
/* 924 */	0x6,		/* FC_SHORT */
			0x6,		/* FC_SHORT */
/* 926 */	0x8,		/* FC_LONG */
			0x8,		/* FC_LONG */
/* 928 */	0x4c,		/* FC_EMBEDDED_COMPLEX */
			0x0,		/* 0 */
/* 930 */	NdrFcShort( 0xfdde ),	/* Offset= -546 (384) */
/* 932 */	0x5c,		/* FC_PAD */
			0x5b,		/* FC_END */
/* 934 */	
			0x13, 0x0,	/* FC_OP */
/* 936 */	NdrFcShort( 0xfeea ),	/* Offset= -278 (658) */
/* 938 */	
			0x13, 0x8,	/* FC_OP [simple_pointer] */
/* 940 */	0x1,		/* FC_BYTE */
			0x5c,		/* FC_PAD */
/* 942 */	
			0x13, 0x8,	/* FC_OP [simple_pointer] */
/* 944 */	0x6,		/* FC_SHORT */
			0x5c,		/* FC_PAD */
/* 946 */	
			0x13, 0x8,	/* FC_OP [simple_pointer] */
/* 948 */	0x8,		/* FC_LONG */
			0x5c,		/* FC_PAD */
/* 950 */	
			0x13, 0x8,	/* FC_OP [simple_pointer] */
/* 952 */	0xb,		/* FC_HYPER */
			0x5c,		/* FC_PAD */
/* 954 */	
			0x13, 0x8,	/* FC_OP [simple_pointer] */
/* 956 */	0xa,		/* FC_FLOAT */
			0x5c,		/* FC_PAD */
/* 958 */	
			0x13, 0x8,	/* FC_OP [simple_pointer] */
/* 960 */	0xc,		/* FC_DOUBLE */
			0x5c,		/* FC_PAD */
/* 962 */	
			0x13, 0x0,	/* FC_OP */
/* 964 */	NdrFcShort( 0xfd70 ),	/* Offset= -656 (308) */
/* 966 */	
			0x13, 0x10,	/* FC_OP [pointer_deref] */
/* 968 */	NdrFcShort( 0xfd72 ),	/* Offset= -654 (314) */
/* 970 */	
			0x13, 0x10,	/* FC_OP [pointer_deref] */
/* 972 */	NdrFcShort( 0xfd88 ),	/* Offset= -632 (340) */
/* 974 */	
			0x13, 0x10,	/* FC_OP [pointer_deref] */
/* 976 */	NdrFcShort( 0xfd96 ),	/* Offset= -618 (358) */
/* 978 */	
			0x13, 0x10,	/* FC_OP [pointer_deref] */
/* 980 */	NdrFcShort( 0xfda4 ),	/* Offset= -604 (376) */
/* 982 */	
			0x13, 0x10,	/* FC_OP [pointer_deref] */
/* 984 */	NdrFcShort( 0x2 ),	/* Offset= 2 (986) */
/* 986 */	
			0x13, 0x0,	/* FC_OP */
/* 988 */	NdrFcShort( 0x14 ),	/* Offset= 20 (1008) */
/* 990 */	
			0x15,		/* FC_STRUCT */
			0x7,		/* 7 */
/* 992 */	NdrFcShort( 0x10 ),	/* 16 */
/* 994 */	0x6,		/* FC_SHORT */
			0x1,		/* FC_BYTE */
/* 996 */	0x1,		/* FC_BYTE */
			0x8,		/* FC_LONG */
/* 998 */	0xb,		/* FC_HYPER */
			0x5b,		/* FC_END */
/* 1000 */	
			0x13, 0x0,	/* FC_OP */
/* 1002 */	NdrFcShort( 0xfff4 ),	/* Offset= -12 (990) */
/* 1004 */	
			0x13, 0x8,	/* FC_OP [simple_pointer] */
/* 1006 */	0x2,		/* FC_CHAR */
			0x5c,		/* FC_PAD */
/* 1008 */	
			0x1a,		/* FC_BOGUS_STRUCT */
			0x7,		/* 7 */
/* 1010 */	NdrFcShort( 0x20 ),	/* 32 */
/* 1012 */	NdrFcShort( 0x0 ),	/* 0 */
/* 1014 */	NdrFcShort( 0x0 ),	/* Offset= 0 (1014) */
/* 1016 */	0x8,		/* FC_LONG */
			0x8,		/* FC_LONG */
/* 1018 */	0x6,		/* FC_SHORT */
			0x6,		/* FC_SHORT */
/* 1020 */	0x6,		/* FC_SHORT */
			0x6,		/* FC_SHORT */
/* 1022 */	0x4c,		/* FC_EMBEDDED_COMPLEX */
			0x0,		/* 0 */
/* 1024 */	NdrFcShort( 0xfc0a ),	/* Offset= -1014 (10) */
/* 1026 */	0x5c,		/* FC_PAD */
			0x5b,		/* FC_END */
/* 1028 */	0xb4,		/* FC_USER_MARSHAL */
			0x83,		/* 131 */
/* 1030 */	NdrFcShort( 0x0 ),	/* 0 */
/* 1032 */	NdrFcShort( 0x10 ),	/* 16 */
/* 1034 */	NdrFcShort( 0x0 ),	/* 0 */
/* 1036 */	NdrFcShort( 0xfbfa ),	/* Offset= -1030 (6) */
/* 1038 */	
			0x12, 0x0,	/* FC_UP */
/* 1040 */	NdrFcShort( 0xffe0 ),	/* Offset= -32 (1008) */
/* 1042 */	0xb4,		/* FC_USER_MARSHAL */
			0x83,		/* 131 */
/* 1044 */	NdrFcShort( 0x0 ),	/* 0 */
/* 1046 */	NdrFcShort( 0x10 ),	/* 16 */
/* 1048 */	NdrFcShort( 0x0 ),	/* 0 */
/* 1050 */	NdrFcShort( 0xfff4 ),	/* Offset= -12 (1038) */
/* 1052 */	
			0x11, 0x0,	/* FC_RP */
/* 1054 */	NdrFcShort( 0xffe6 ),	/* Offset= -26 (1028) */

			0x0
        }
    };

static const USER_MARSHAL_ROUTINE_QUADRUPLE UserMarshalRoutines[ WIRE_MARSHAL_TABLE_SIZE ] = 
        {
            
            {
            VARIANT_UserSize
            ,VARIANT_UserMarshal
            ,VARIANT_UserUnmarshal
            ,VARIANT_UserFree
            }

        };



/* Object interface: IUnknown, ver. 0.0,
   GUID={0x00000000,0x0000,0x0000,{0xC0,0x00,0x00,0x00,0x00,0x00,0x00,0x46}} */


/* Object interface: IDispatch, ver. 0.0,
   GUID={0x00020400,0x0000,0x0000,{0xC0,0x00,0x00,0x00,0x00,0x00,0x00,0x46}} */


/* Object interface: IMWUtil, ver. 0.0,
   GUID={0xC47EA90E,0x56D1,0x11d5,{0xB1,0x59,0x00,0xD0,0xB7,0xBA,0x75,0x44}} */

#pragma code_seg(".orpc")
static const unsigned short IMWUtil_FormatStringOffsetTable[] =
    {
    (unsigned short) -1,
    (unsigned short) -1,
    (unsigned short) -1,
    (unsigned short) -1,
    0,
    228,
    468,
    504
    };

static const MIDL_STUBLESS_PROXY_INFO IMWUtil_ProxyInfo =
    {
    &Object_StubDesc,
    __MIDL_ProcFormatString.Format,
    &IMWUtil_FormatStringOffsetTable[-3],
    0,
    0,
    0
    };


static const MIDL_SERVER_INFO IMWUtil_ServerInfo = 
    {
    &Object_StubDesc,
    0,
    __MIDL_ProcFormatString.Format,
    &IMWUtil_FormatStringOffsetTable[-3],
    0,
    0,
    0,
    0};
CINTERFACE_PROXY_VTABLE(11) _IMWUtilProxyVtbl = 
{
    &IMWUtil_ProxyInfo,
    &IID_IMWUtil,
    IUnknown_QueryInterface_Proxy,
    IUnknown_AddRef_Proxy,
    IUnknown_Release_Proxy ,
    0 /* (void *) (INT_PTR) -1 /* IDispatch::GetTypeInfoCount */ ,
    0 /* (void *) (INT_PTR) -1 /* IDispatch::GetTypeInfo */ ,
    0 /* (void *) (INT_PTR) -1 /* IDispatch::GetIDsOfNames */ ,
    0 /* IDispatch_Invoke_Proxy */ ,
    (void *) (INT_PTR) -1 /* IMWUtil::MWPack */ ,
    (void *) (INT_PTR) -1 /* IMWUtil::MWUnpack */ ,
    (void *) (INT_PTR) -1 /* IMWUtil::MWDate2VariantDate */ ,
    (void *) (INT_PTR) -1 /* IMWUtil::MWInitApplication */
};


static const PRPC_STUB_FUNCTION IMWUtil_table[] =
{
    STUB_FORWARDING_FUNCTION,
    STUB_FORWARDING_FUNCTION,
    STUB_FORWARDING_FUNCTION,
    STUB_FORWARDING_FUNCTION,
    NdrStubCall2,
    NdrStubCall2,
    NdrStubCall2,
    NdrStubCall2
};

CInterfaceStubVtbl _IMWUtilStubVtbl =
{
    &IID_IMWUtil,
    &IMWUtil_ServerInfo,
    11,
    &IMWUtil_table[-3],
    CStdStubBuffer_DELEGATING_METHODS
};

static const MIDL_STUB_DESC Object_StubDesc = 
    {
    0,
    NdrOleAllocate,
    NdrOleFree,
    0,
    0,
    0,
    0,
    0,
    __MIDL_TypeFormatString.Format,
    1, /* -error bounds_check flag */
    0x50002, /* Ndr library version */
    0,
    0x600016e, /* MIDL Version 6.0.366 */
    0,
    UserMarshalRoutines,
    0,  /* notify & notify_flag routine table */
    0x1, /* MIDL flag */
    0, /* cs routines */
    0,   /* proxy/server info */
    0   /* Reserved5 */
    };

const CInterfaceProxyVtbl * _mwcomutil_ProxyVtblList[] = 
{
    ( CInterfaceProxyVtbl *) &_IMWUtilProxyVtbl,
    0
};

const CInterfaceStubVtbl * _mwcomutil_StubVtblList[] = 
{
    ( CInterfaceStubVtbl *) &_IMWUtilStubVtbl,
    0
};

PCInterfaceName const _mwcomutil_InterfaceNamesList[] = 
{
    "IMWUtil",
    0
};

const IID *  _mwcomutil_BaseIIDList[] = 
{
    &IID_IDispatch,
    0
};


#define _mwcomutil_CHECK_IID(n)	IID_GENERIC_CHECK_IID( _mwcomutil, pIID, n)

int __stdcall _mwcomutil_IID_Lookup( const IID * pIID, int * pIndex )
{
    
    if(!_mwcomutil_CHECK_IID(0))
        {
        *pIndex = 0;
        return 1;
        }

    return 0;
}

const ExtendedProxyFileInfo mwcomutil_ProxyFileInfo = 
{
    (PCInterfaceProxyVtblList *) & _mwcomutil_ProxyVtblList,
    (PCInterfaceStubVtblList *) & _mwcomutil_StubVtblList,
    (const PCInterfaceName * ) & _mwcomutil_InterfaceNamesList,
    (const IID ** ) & _mwcomutil_BaseIIDList,
    & _mwcomutil_IID_Lookup, 
    1,
    2,
    0, /* table of [async_uuid] interfaces */
    0, /* Filler1 */
    0, /* Filler2 */
    0  /* Filler3 */
};
#pragma optimize("", on )
#if _MSC_VER >= 1200
#pragma warning(pop)
#endif


#endif /* !defined(_M_IA64) && !defined(_M_AMD64)*/


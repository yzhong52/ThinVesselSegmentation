

/* this ALWAYS GENERATED file contains the definitions for the interfaces */


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


/* verify that the <rpcndr.h> version is high enough to compile this file*/
#ifndef __REQUIRED_RPCNDR_H_VERSION__
#define __REQUIRED_RPCNDR_H_VERSION__ 475
#endif

#include "rpc.h"
#include "rpcndr.h"

#ifndef __RPCNDR_H_VERSION__
#error this stub requires an updated version of <rpcndr.h>
#endif // __RPCNDR_H_VERSION__

#ifndef COM_NO_WINDOWS_H
#include "windows.h"
#include "ole2.h"
#endif /*COM_NO_WINDOWS_H*/

#ifndef __mwcomutil_h__
#define __mwcomutil_h__

#if defined(_MSC_VER) && (_MSC_VER >= 1020)
#pragma once
#endif

/* Forward Declarations */ 

#ifndef __IMWUtil_FWD_DEFINED__
#define __IMWUtil_FWD_DEFINED__
typedef interface IMWUtil IMWUtil;
#endif 	/* __IMWUtil_FWD_DEFINED__ */


#ifndef __MWField_FWD_DEFINED__
#define __MWField_FWD_DEFINED__

#ifdef __cplusplus
typedef class MWField MWField;
#else
typedef struct MWField MWField;
#endif /* __cplusplus */

#endif 	/* __MWField_FWD_DEFINED__ */


#ifndef __MWStruct_FWD_DEFINED__
#define __MWStruct_FWD_DEFINED__

#ifdef __cplusplus
typedef class MWStruct MWStruct;
#else
typedef struct MWStruct MWStruct;
#endif /* __cplusplus */

#endif 	/* __MWStruct_FWD_DEFINED__ */


#ifndef __MWComplex_FWD_DEFINED__
#define __MWComplex_FWD_DEFINED__

#ifdef __cplusplus
typedef class MWComplex MWComplex;
#else
typedef struct MWComplex MWComplex;
#endif /* __cplusplus */

#endif 	/* __MWComplex_FWD_DEFINED__ */


#ifndef __MWSparse_FWD_DEFINED__
#define __MWSparse_FWD_DEFINED__

#ifdef __cplusplus
typedef class MWSparse MWSparse;
#else
typedef struct MWSparse MWSparse;
#endif /* __cplusplus */

#endif 	/* __MWSparse_FWD_DEFINED__ */


#ifndef __MWArg_FWD_DEFINED__
#define __MWArg_FWD_DEFINED__

#ifdef __cplusplus
typedef class MWArg MWArg;
#else
typedef struct MWArg MWArg;
#endif /* __cplusplus */

#endif 	/* __MWArg_FWD_DEFINED__ */


#ifndef __MWArrayFormatFlags_FWD_DEFINED__
#define __MWArrayFormatFlags_FWD_DEFINED__

#ifdef __cplusplus
typedef class MWArrayFormatFlags MWArrayFormatFlags;
#else
typedef struct MWArrayFormatFlags MWArrayFormatFlags;
#endif /* __cplusplus */

#endif 	/* __MWArrayFormatFlags_FWD_DEFINED__ */


#ifndef __MWDataConversionFlags_FWD_DEFINED__
#define __MWDataConversionFlags_FWD_DEFINED__

#ifdef __cplusplus
typedef class MWDataConversionFlags MWDataConversionFlags;
#else
typedef struct MWDataConversionFlags MWDataConversionFlags;
#endif /* __cplusplus */

#endif 	/* __MWDataConversionFlags_FWD_DEFINED__ */


#ifndef __MWUtil_FWD_DEFINED__
#define __MWUtil_FWD_DEFINED__

#ifdef __cplusplus
typedef class MWUtil MWUtil;
#else
typedef struct MWUtil MWUtil;
#endif /* __cplusplus */

#endif 	/* __MWUtil_FWD_DEFINED__ */


#ifndef __MWFlags_FWD_DEFINED__
#define __MWFlags_FWD_DEFINED__

#ifdef __cplusplus
typedef class MWFlags MWFlags;
#else
typedef struct MWFlags MWFlags;
#endif /* __cplusplus */

#endif 	/* __MWFlags_FWD_DEFINED__ */


/* header files for imported files */
#include "oaidl.h"
#include "ocidl.h"
#include "mwcomtypes.h"

#ifdef __cplusplus
extern "C"{
#endif 

void * __RPC_USER MIDL_user_allocate(size_t);
void __RPC_USER MIDL_user_free( void * ); 

#ifndef __IMWUtil_INTERFACE_DEFINED__
#define __IMWUtil_INTERFACE_DEFINED__

/* interface IMWUtil */
/* [unique][helpstring][dual][uuid][object] */ 


EXTERN_C const IID IID_IMWUtil;

#if defined(__cplusplus) && !defined(CINTERFACE)
    
    MIDL_INTERFACE("C47EA90E-56D1-11d5-B159-00D0B7BA7544")
    IMWUtil : public IDispatch
    {
    public:
        virtual /* [helpstring][id] */ HRESULT STDMETHODCALLTYPE MWPack( 
            /* [out] */ VARIANT *pVarArg,
            /* [optional][in] */ VARIANT Var0,
            /* [optional][in] */ VARIANT Var1,
            /* [optional][in] */ VARIANT Var2,
            /* [optional][in] */ VARIANT Var3,
            /* [optional][in] */ VARIANT Var4,
            /* [optional][in] */ VARIANT Var5,
            /* [optional][in] */ VARIANT Var6,
            /* [optional][in] */ VARIANT Var7,
            /* [optional][in] */ VARIANT Var8,
            /* [optional][in] */ VARIANT Var9,
            /* [optional][in] */ VARIANT Var10,
            /* [optional][in] */ VARIANT Var11,
            /* [optional][in] */ VARIANT Var12,
            /* [optional][in] */ VARIANT Var13,
            /* [optional][in] */ VARIANT Var14,
            /* [optional][in] */ VARIANT Var15,
            /* [optional][in] */ VARIANT Var16,
            /* [optional][in] */ VARIANT Var17,
            /* [optional][in] */ VARIANT Var18,
            /* [optional][in] */ VARIANT Var19,
            /* [optional][in] */ VARIANT Var20,
            /* [optional][in] */ VARIANT Var21,
            /* [optional][in] */ VARIANT Var22,
            /* [optional][in] */ VARIANT Var23,
            /* [optional][in] */ VARIANT Var24,
            /* [optional][in] */ VARIANT Var25,
            /* [optional][in] */ VARIANT Var26,
            /* [optional][in] */ VARIANT Var27,
            /* [optional][in] */ VARIANT Var28,
            /* [optional][in] */ VARIANT Var29,
            /* [optional][in] */ VARIANT Var30,
            /* [optional][in] */ VARIANT Var31) = 0;
        
        virtual /* [helpstring][id] */ HRESULT STDMETHODCALLTYPE MWUnpack( 
            /* [in] */ VARIANT VarArg,
            /* [defaultvalue][in] */ long nStartAt,
            /* [defaultvalue][in] */ VARIANT_BOOL bAutoResize,
            /* [optional][out][in] */ VARIANT *pVar0,
            /* [optional][out][in] */ VARIANT *pVar1,
            /* [optional][out][in] */ VARIANT *pVar2,
            /* [optional][out][in] */ VARIANT *pVar3,
            /* [optional][out][in] */ VARIANT *pVar4,
            /* [optional][out][in] */ VARIANT *pVar5,
            /* [optional][out][in] */ VARIANT *pVar6,
            /* [optional][out][in] */ VARIANT *pVar7,
            /* [optional][out][in] */ VARIANT *pVar8,
            /* [optional][out][in] */ VARIANT *pVar9,
            /* [optional][out][in] */ VARIANT *pVar10,
            /* [optional][out][in] */ VARIANT *pVar11,
            /* [optional][out][in] */ VARIANT *pVar12,
            /* [optional][out][in] */ VARIANT *pVar13,
            /* [optional][out][in] */ VARIANT *pVar14,
            /* [optional][out][in] */ VARIANT *pVar15,
            /* [optional][out][in] */ VARIANT *pVar16,
            /* [optional][out][in] */ VARIANT *pVar17,
            /* [optional][out][in] */ VARIANT *pVar18,
            /* [optional][out][in] */ VARIANT *pVar19,
            /* [optional][out][in] */ VARIANT *pVar20,
            /* [optional][out][in] */ VARIANT *pVar21,
            /* [optional][out][in] */ VARIANT *pVar22,
            /* [optional][out][in] */ VARIANT *pVar23,
            /* [optional][out][in] */ VARIANT *pVar24,
            /* [optional][out][in] */ VARIANT *pVar25,
            /* [optional][out][in] */ VARIANT *pVar26,
            /* [optional][out][in] */ VARIANT *pVar27,
            /* [optional][out][in] */ VARIANT *pVar28,
            /* [optional][out][in] */ VARIANT *pVar29,
            /* [optional][out][in] */ VARIANT *pVar30,
            /* [optional][out][in] */ VARIANT *pVar31) = 0;
        
        virtual /* [helpstring][id] */ HRESULT STDMETHODCALLTYPE MWDate2VariantDate( 
            /* [out][in] */ VARIANT *pVar) = 0;
        
        virtual /* [helpstring][id] */ HRESULT STDMETHODCALLTYPE MWInitApplication( 
            /* [in] */ IDispatch *pApp) = 0;
        
    };
    
#else 	/* C style interface */

    typedef struct IMWUtilVtbl
    {
        BEGIN_INTERFACE
        
        HRESULT ( STDMETHODCALLTYPE *QueryInterface )( 
            IMWUtil * This,
            /* [in] */ REFIID riid,
            /* [iid_is][out] */ void **ppvObject);
        
        ULONG ( STDMETHODCALLTYPE *AddRef )( 
            IMWUtil * This);
        
        ULONG ( STDMETHODCALLTYPE *Release )( 
            IMWUtil * This);
        
        HRESULT ( STDMETHODCALLTYPE *GetTypeInfoCount )( 
            IMWUtil * This,
            /* [out] */ UINT *pctinfo);
        
        HRESULT ( STDMETHODCALLTYPE *GetTypeInfo )( 
            IMWUtil * This,
            /* [in] */ UINT iTInfo,
            /* [in] */ LCID lcid,
            /* [out] */ ITypeInfo **ppTInfo);
        
        HRESULT ( STDMETHODCALLTYPE *GetIDsOfNames )( 
            IMWUtil * This,
            /* [in] */ REFIID riid,
            /* [size_is][in] */ LPOLESTR *rgszNames,
            /* [in] */ UINT cNames,
            /* [in] */ LCID lcid,
            /* [size_is][out] */ DISPID *rgDispId);
        
        /* [local] */ HRESULT ( STDMETHODCALLTYPE *Invoke )( 
            IMWUtil * This,
            /* [in] */ DISPID dispIdMember,
            /* [in] */ REFIID riid,
            /* [in] */ LCID lcid,
            /* [in] */ WORD wFlags,
            /* [out][in] */ DISPPARAMS *pDispParams,
            /* [out] */ VARIANT *pVarResult,
            /* [out] */ EXCEPINFO *pExcepInfo,
            /* [out] */ UINT *puArgErr);
        
        /* [helpstring][id] */ HRESULT ( STDMETHODCALLTYPE *MWPack )( 
            IMWUtil * This,
            /* [out] */ VARIANT *pVarArg,
            /* [optional][in] */ VARIANT Var0,
            /* [optional][in] */ VARIANT Var1,
            /* [optional][in] */ VARIANT Var2,
            /* [optional][in] */ VARIANT Var3,
            /* [optional][in] */ VARIANT Var4,
            /* [optional][in] */ VARIANT Var5,
            /* [optional][in] */ VARIANT Var6,
            /* [optional][in] */ VARIANT Var7,
            /* [optional][in] */ VARIANT Var8,
            /* [optional][in] */ VARIANT Var9,
            /* [optional][in] */ VARIANT Var10,
            /* [optional][in] */ VARIANT Var11,
            /* [optional][in] */ VARIANT Var12,
            /* [optional][in] */ VARIANT Var13,
            /* [optional][in] */ VARIANT Var14,
            /* [optional][in] */ VARIANT Var15,
            /* [optional][in] */ VARIANT Var16,
            /* [optional][in] */ VARIANT Var17,
            /* [optional][in] */ VARIANT Var18,
            /* [optional][in] */ VARIANT Var19,
            /* [optional][in] */ VARIANT Var20,
            /* [optional][in] */ VARIANT Var21,
            /* [optional][in] */ VARIANT Var22,
            /* [optional][in] */ VARIANT Var23,
            /* [optional][in] */ VARIANT Var24,
            /* [optional][in] */ VARIANT Var25,
            /* [optional][in] */ VARIANT Var26,
            /* [optional][in] */ VARIANT Var27,
            /* [optional][in] */ VARIANT Var28,
            /* [optional][in] */ VARIANT Var29,
            /* [optional][in] */ VARIANT Var30,
            /* [optional][in] */ VARIANT Var31);
        
        /* [helpstring][id] */ HRESULT ( STDMETHODCALLTYPE *MWUnpack )( 
            IMWUtil * This,
            /* [in] */ VARIANT VarArg,
            /* [defaultvalue][in] */ long nStartAt,
            /* [defaultvalue][in] */ VARIANT_BOOL bAutoResize,
            /* [optional][out][in] */ VARIANT *pVar0,
            /* [optional][out][in] */ VARIANT *pVar1,
            /* [optional][out][in] */ VARIANT *pVar2,
            /* [optional][out][in] */ VARIANT *pVar3,
            /* [optional][out][in] */ VARIANT *pVar4,
            /* [optional][out][in] */ VARIANT *pVar5,
            /* [optional][out][in] */ VARIANT *pVar6,
            /* [optional][out][in] */ VARIANT *pVar7,
            /* [optional][out][in] */ VARIANT *pVar8,
            /* [optional][out][in] */ VARIANT *pVar9,
            /* [optional][out][in] */ VARIANT *pVar10,
            /* [optional][out][in] */ VARIANT *pVar11,
            /* [optional][out][in] */ VARIANT *pVar12,
            /* [optional][out][in] */ VARIANT *pVar13,
            /* [optional][out][in] */ VARIANT *pVar14,
            /* [optional][out][in] */ VARIANT *pVar15,
            /* [optional][out][in] */ VARIANT *pVar16,
            /* [optional][out][in] */ VARIANT *pVar17,
            /* [optional][out][in] */ VARIANT *pVar18,
            /* [optional][out][in] */ VARIANT *pVar19,
            /* [optional][out][in] */ VARIANT *pVar20,
            /* [optional][out][in] */ VARIANT *pVar21,
            /* [optional][out][in] */ VARIANT *pVar22,
            /* [optional][out][in] */ VARIANT *pVar23,
            /* [optional][out][in] */ VARIANT *pVar24,
            /* [optional][out][in] */ VARIANT *pVar25,
            /* [optional][out][in] */ VARIANT *pVar26,
            /* [optional][out][in] */ VARIANT *pVar27,
            /* [optional][out][in] */ VARIANT *pVar28,
            /* [optional][out][in] */ VARIANT *pVar29,
            /* [optional][out][in] */ VARIANT *pVar30,
            /* [optional][out][in] */ VARIANT *pVar31);
        
        /* [helpstring][id] */ HRESULT ( STDMETHODCALLTYPE *MWDate2VariantDate )( 
            IMWUtil * This,
            /* [out][in] */ VARIANT *pVar);
        
        /* [helpstring][id] */ HRESULT ( STDMETHODCALLTYPE *MWInitApplication )( 
            IMWUtil * This,
            /* [in] */ IDispatch *pApp);
        
        END_INTERFACE
    } IMWUtilVtbl;

    interface IMWUtil
    {
        CONST_VTBL struct IMWUtilVtbl *lpVtbl;
    };

    

#ifdef COBJMACROS


#define IMWUtil_QueryInterface(This,riid,ppvObject)	\
    (This)->lpVtbl -> QueryInterface(This,riid,ppvObject)

#define IMWUtil_AddRef(This)	\
    (This)->lpVtbl -> AddRef(This)

#define IMWUtil_Release(This)	\
    (This)->lpVtbl -> Release(This)


#define IMWUtil_GetTypeInfoCount(This,pctinfo)	\
    (This)->lpVtbl -> GetTypeInfoCount(This,pctinfo)

#define IMWUtil_GetTypeInfo(This,iTInfo,lcid,ppTInfo)	\
    (This)->lpVtbl -> GetTypeInfo(This,iTInfo,lcid,ppTInfo)

#define IMWUtil_GetIDsOfNames(This,riid,rgszNames,cNames,lcid,rgDispId)	\
    (This)->lpVtbl -> GetIDsOfNames(This,riid,rgszNames,cNames,lcid,rgDispId)

#define IMWUtil_Invoke(This,dispIdMember,riid,lcid,wFlags,pDispParams,pVarResult,pExcepInfo,puArgErr)	\
    (This)->lpVtbl -> Invoke(This,dispIdMember,riid,lcid,wFlags,pDispParams,pVarResult,pExcepInfo,puArgErr)


#define IMWUtil_MWPack(This,pVarArg,Var0,Var1,Var2,Var3,Var4,Var5,Var6,Var7,Var8,Var9,Var10,Var11,Var12,Var13,Var14,Var15,Var16,Var17,Var18,Var19,Var20,Var21,Var22,Var23,Var24,Var25,Var26,Var27,Var28,Var29,Var30,Var31)	\
    (This)->lpVtbl -> MWPack(This,pVarArg,Var0,Var1,Var2,Var3,Var4,Var5,Var6,Var7,Var8,Var9,Var10,Var11,Var12,Var13,Var14,Var15,Var16,Var17,Var18,Var19,Var20,Var21,Var22,Var23,Var24,Var25,Var26,Var27,Var28,Var29,Var30,Var31)

#define IMWUtil_MWUnpack(This,VarArg,nStartAt,bAutoResize,pVar0,pVar1,pVar2,pVar3,pVar4,pVar5,pVar6,pVar7,pVar8,pVar9,pVar10,pVar11,pVar12,pVar13,pVar14,pVar15,pVar16,pVar17,pVar18,pVar19,pVar20,pVar21,pVar22,pVar23,pVar24,pVar25,pVar26,pVar27,pVar28,pVar29,pVar30,pVar31)	\
    (This)->lpVtbl -> MWUnpack(This,VarArg,nStartAt,bAutoResize,pVar0,pVar1,pVar2,pVar3,pVar4,pVar5,pVar6,pVar7,pVar8,pVar9,pVar10,pVar11,pVar12,pVar13,pVar14,pVar15,pVar16,pVar17,pVar18,pVar19,pVar20,pVar21,pVar22,pVar23,pVar24,pVar25,pVar26,pVar27,pVar28,pVar29,pVar30,pVar31)

#define IMWUtil_MWDate2VariantDate(This,pVar)	\
    (This)->lpVtbl -> MWDate2VariantDate(This,pVar)

#define IMWUtil_MWInitApplication(This,pApp)	\
    (This)->lpVtbl -> MWInitApplication(This,pApp)

#endif /* COBJMACROS */


#endif 	/* C style interface */



/* [helpstring][id] */ HRESULT STDMETHODCALLTYPE IMWUtil_MWPack_Proxy( 
    IMWUtil * This,
    /* [out] */ VARIANT *pVarArg,
    /* [optional][in] */ VARIANT Var0,
    /* [optional][in] */ VARIANT Var1,
    /* [optional][in] */ VARIANT Var2,
    /* [optional][in] */ VARIANT Var3,
    /* [optional][in] */ VARIANT Var4,
    /* [optional][in] */ VARIANT Var5,
    /* [optional][in] */ VARIANT Var6,
    /* [optional][in] */ VARIANT Var7,
    /* [optional][in] */ VARIANT Var8,
    /* [optional][in] */ VARIANT Var9,
    /* [optional][in] */ VARIANT Var10,
    /* [optional][in] */ VARIANT Var11,
    /* [optional][in] */ VARIANT Var12,
    /* [optional][in] */ VARIANT Var13,
    /* [optional][in] */ VARIANT Var14,
    /* [optional][in] */ VARIANT Var15,
    /* [optional][in] */ VARIANT Var16,
    /* [optional][in] */ VARIANT Var17,
    /* [optional][in] */ VARIANT Var18,
    /* [optional][in] */ VARIANT Var19,
    /* [optional][in] */ VARIANT Var20,
    /* [optional][in] */ VARIANT Var21,
    /* [optional][in] */ VARIANT Var22,
    /* [optional][in] */ VARIANT Var23,
    /* [optional][in] */ VARIANT Var24,
    /* [optional][in] */ VARIANT Var25,
    /* [optional][in] */ VARIANT Var26,
    /* [optional][in] */ VARIANT Var27,
    /* [optional][in] */ VARIANT Var28,
    /* [optional][in] */ VARIANT Var29,
    /* [optional][in] */ VARIANT Var30,
    /* [optional][in] */ VARIANT Var31);


void __RPC_STUB IMWUtil_MWPack_Stub(
    IRpcStubBuffer *This,
    IRpcChannelBuffer *_pRpcChannelBuffer,
    PRPC_MESSAGE _pRpcMessage,
    DWORD *_pdwStubPhase);


/* [helpstring][id] */ HRESULT STDMETHODCALLTYPE IMWUtil_MWUnpack_Proxy( 
    IMWUtil * This,
    /* [in] */ VARIANT VarArg,
    /* [defaultvalue][in] */ long nStartAt,
    /* [defaultvalue][in] */ VARIANT_BOOL bAutoResize,
    /* [optional][out][in] */ VARIANT *pVar0,
    /* [optional][out][in] */ VARIANT *pVar1,
    /* [optional][out][in] */ VARIANT *pVar2,
    /* [optional][out][in] */ VARIANT *pVar3,
    /* [optional][out][in] */ VARIANT *pVar4,
    /* [optional][out][in] */ VARIANT *pVar5,
    /* [optional][out][in] */ VARIANT *pVar6,
    /* [optional][out][in] */ VARIANT *pVar7,
    /* [optional][out][in] */ VARIANT *pVar8,
    /* [optional][out][in] */ VARIANT *pVar9,
    /* [optional][out][in] */ VARIANT *pVar10,
    /* [optional][out][in] */ VARIANT *pVar11,
    /* [optional][out][in] */ VARIANT *pVar12,
    /* [optional][out][in] */ VARIANT *pVar13,
    /* [optional][out][in] */ VARIANT *pVar14,
    /* [optional][out][in] */ VARIANT *pVar15,
    /* [optional][out][in] */ VARIANT *pVar16,
    /* [optional][out][in] */ VARIANT *pVar17,
    /* [optional][out][in] */ VARIANT *pVar18,
    /* [optional][out][in] */ VARIANT *pVar19,
    /* [optional][out][in] */ VARIANT *pVar20,
    /* [optional][out][in] */ VARIANT *pVar21,
    /* [optional][out][in] */ VARIANT *pVar22,
    /* [optional][out][in] */ VARIANT *pVar23,
    /* [optional][out][in] */ VARIANT *pVar24,
    /* [optional][out][in] */ VARIANT *pVar25,
    /* [optional][out][in] */ VARIANT *pVar26,
    /* [optional][out][in] */ VARIANT *pVar27,
    /* [optional][out][in] */ VARIANT *pVar28,
    /* [optional][out][in] */ VARIANT *pVar29,
    /* [optional][out][in] */ VARIANT *pVar30,
    /* [optional][out][in] */ VARIANT *pVar31);


void __RPC_STUB IMWUtil_MWUnpack_Stub(
    IRpcStubBuffer *This,
    IRpcChannelBuffer *_pRpcChannelBuffer,
    PRPC_MESSAGE _pRpcMessage,
    DWORD *_pdwStubPhase);


/* [helpstring][id] */ HRESULT STDMETHODCALLTYPE IMWUtil_MWDate2VariantDate_Proxy( 
    IMWUtil * This,
    /* [out][in] */ VARIANT *pVar);


void __RPC_STUB IMWUtil_MWDate2VariantDate_Stub(
    IRpcStubBuffer *This,
    IRpcChannelBuffer *_pRpcChannelBuffer,
    PRPC_MESSAGE _pRpcMessage,
    DWORD *_pdwStubPhase);


/* [helpstring][id] */ HRESULT STDMETHODCALLTYPE IMWUtil_MWInitApplication_Proxy( 
    IMWUtil * This,
    /* [in] */ IDispatch *pApp);


void __RPC_STUB IMWUtil_MWInitApplication_Stub(
    IRpcStubBuffer *This,
    IRpcChannelBuffer *_pRpcChannelBuffer,
    PRPC_MESSAGE _pRpcMessage,
    DWORD *_pdwStubPhase);



#endif 	/* __IMWUtil_INTERFACE_DEFINED__ */



#ifndef __MWComUtil_LIBRARY_DEFINED__
#define __MWComUtil_LIBRARY_DEFINED__

/* library MWComUtil */
/* [helpstring][version][uuid] */ 


EXTERN_C const IID LIBID_MWComUtil;

EXTERN_C const CLSID CLSID_MWField;

#ifdef __cplusplus

class DECLSPEC_UUID("0A145DE2-5EED-4C66-9443-48720472CEEC")
MWField;
#endif

EXTERN_C const CLSID CLSID_MWStruct;

#ifdef __cplusplus

class DECLSPEC_UUID("E72CDF78-C3E2-41FA-89A1-004A25371DA5")
MWStruct;
#endif

EXTERN_C const CLSID CLSID_MWComplex;

#ifdef __cplusplus

class DECLSPEC_UUID("E53E3B4B-E24C-449E-91AC-2EE1B816A831")
MWComplex;
#endif

EXTERN_C const CLSID CLSID_MWSparse;

#ifdef __cplusplus

class DECLSPEC_UUID("3E347B31-48C3-427F-BE41-71E7BC77333A")
MWSparse;
#endif

EXTERN_C const CLSID CLSID_MWArg;

#ifdef __cplusplus

class DECLSPEC_UUID("6C7F426D-A7DE-4EA6-8715-A88E6C2F735C")
MWArg;
#endif

EXTERN_C const CLSID CLSID_MWArrayFormatFlags;

#ifdef __cplusplus

class DECLSPEC_UUID("A0CFC286-9018-411D-8317-A20F16604F5F")
MWArrayFormatFlags;
#endif

EXTERN_C const CLSID CLSID_MWDataConversionFlags;

#ifdef __cplusplus

class DECLSPEC_UUID("A79AD773-380B-451C-9DF1-85355C3E40BD")
MWDataConversionFlags;
#endif

EXTERN_C const CLSID CLSID_MWUtil;

#ifdef __cplusplus

class DECLSPEC_UUID("CE30EF50-DB22-4934-B4C2-C08EF205EE55")
MWUtil;
#endif

EXTERN_C const CLSID CLSID_MWFlags;

#ifdef __cplusplus

class DECLSPEC_UUID("E31161EC-1E33-458C-8CAD-1415165E3F53")
MWFlags;
#endif
#endif /* __MWComUtil_LIBRARY_DEFINED__ */

/* Additional Prototypes for ALL interfaces */

unsigned long             __RPC_USER  VARIANT_UserSize(     unsigned long *, unsigned long            , VARIANT * ); 
unsigned char * __RPC_USER  VARIANT_UserMarshal(  unsigned long *, unsigned char *, VARIANT * ); 
unsigned char * __RPC_USER  VARIANT_UserUnmarshal(unsigned long *, unsigned char *, VARIANT * ); 
void                      __RPC_USER  VARIANT_UserFree(     unsigned long *, VARIANT * ); 

/* end of Additional Prototypes */

#ifdef __cplusplus
}
#endif

#endif



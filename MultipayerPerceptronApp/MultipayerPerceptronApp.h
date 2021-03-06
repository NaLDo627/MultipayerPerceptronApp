
// MultipayerPerceptronApp.h : main header file for the PROJECT_NAME application
//

#pragma once

#ifndef __AFXWIN_H__
	#error "include 'stdafx.h' before including this file for PCH"
#endif

#include "resource.h"		// main symbols
#include "GridCtrl\GridCtrl.h"
#include <math.h>
#include "MultiPayerPerceptronNetwork.h"

// CMultipayerPerceptronAppApp:
// See MultipayerPerceptronApp.cpp for the implementation of this class
//

class CMultipayerPerceptronAppApp : public CWinApp
{
public:
	CMultipayerPerceptronAppApp();

// Overrides
public:
	virtual BOOL InitInstance();

// Implementation

	DECLARE_MESSAGE_MAP()
};

extern CMultipayerPerceptronAppApp theApp;


// MultipayerPerceptronAppDlg.h : header file
//

#pragma once
#include "afxcmn.h"

#define PATTERN_FILE_NAME	"pattern2.txt"
#define ROW_COL				11


// CMultipayerPerceptronAppDlg dialog
class CMultipayerPerceptronAppDlg : public CDialog,
									public CGridEventHandler
{
// Construction
public:
	CMultipayerPerceptronAppDlg(CWnd* pParent = NULL);	// standard constructor

// Dialog Data
#ifdef AFX_DESIGN_TIME
	enum { IDD = IDD_MULTIPAYERPERCEPTRONAPP_DIALOG };
#endif

	protected:
	virtual void DoDataExchange(CDataExchange* pDX);	// DDX/DDV support


// Implementation
protected:
	HICON m_hIcon;

	// Generated message map functions
	virtual BOOL OnInitDialog();
	afx_msg void OnSysCommand(UINT nID, LPARAM lParam);
	afx_msg void OnPaint();
	afx_msg HCURSOR OnQueryDragIcon();
	DECLARE_MESSAGE_MAP()

/*
	DECLARE_DHTML_EVENT_MAP()*/

	// Grid event handler
	virtual VOID OnGridClick(INT a_nRow, INT a_nCol);

private:
	// GUI
	VOID InitGrid();

	// Variables
	CGridCtrl	m_grdInput;
	CGridCtrl	m_grdOutput;
	CMultiPayerPerceptronNetwork* m_pNetwork;


public: // Automatically generated
	afx_msg void OnBnClickedBtnInput();
	afx_msg void OnBnClickedBtnReset();
	afx_msg void OnClose();
};

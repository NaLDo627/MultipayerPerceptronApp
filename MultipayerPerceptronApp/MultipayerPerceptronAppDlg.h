
// MultipayerPerceptronAppDlg.h : header file
//

#pragma once


// CMultipayerPerceptronAppDlg dialog
class CMultipayerPerceptronAppDlg : public CDialog
{
// Construction
public:
	CMultipayerPerceptronAppDlg(CWnd* pParent = NULL);	// standard constructor

// Dialog Data
#ifdef AFX_DESIGN_TIME
	enum { IDD = IDD_MULTIPAYERPERCEPTRONAPP_DIALOG, IDH = IDR_HTML_MULTIPAYERPERCEPTRONAPP_DIALOG };
#endif

	protected:
	virtual void DoDataExchange(CDataExchange* pDX);	// DDX/DDV support

	HRESULT OnButtonOK(IHTMLElement *pElement);
	HRESULT OnButtonCancel(IHTMLElement *pElement);

// Implementation
protected:
	HICON m_hIcon;

	// Generated message map functions
	virtual BOOL OnInitDialog();
	afx_msg void OnSysCommand(UINT nID, LPARAM lParam);
	afx_msg void OnPaint();
	afx_msg HCURSOR OnQueryDragIcon();
	DECLARE_MESSAGE_MAP()
	DECLARE_DHTML_EVENT_MAP()
};

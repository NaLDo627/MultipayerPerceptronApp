
// MultipayerPerceptronAppDlg.cpp : implementation file
//

#include "stdafx.h"
#include "MultipayerPerceptronApp.h"
#include "MultipayerPerceptronAppDlg.h"
#include "afxdialogex.h"

#ifdef _DEBUG
#define new DEBUG_NEW
#endif

// CAboutDlg dialog used for App About

class CAboutDlg : public CDialogEx
{
public:
	CAboutDlg();

// Dialog Data
#ifdef AFX_DESIGN_TIME
	enum { IDD = IDD_ABOUTBOX };
#endif

	protected:
	virtual void DoDataExchange(CDataExchange* pDX);    // DDX/DDV support

// Implementation
protected:
	DECLARE_MESSAGE_MAP()
};

CAboutDlg::CAboutDlg() : CDialogEx(IDD_ABOUTBOX)
{
}

void CAboutDlg::DoDataExchange(CDataExchange* pDX)
{
	CDialogEx::DoDataExchange(pDX);
}

BEGIN_MESSAGE_MAP(CAboutDlg, CDialogEx)
END_MESSAGE_MAP()


// CMultipayerPerceptronAppDlg dialog
//
//BEGIN_DHTML_EVENT_MAP(CMultipayerPerceptronAppDlg)
//	DHTML_EVENT_ONCLICK(_T("ButtonOK"), OnButtonOK)
//	DHTML_EVENT_ONCLICK(_T("ButtonCancel"), OnButtonCancel)
//END_DHTML_EVENT_MAP()


CMultipayerPerceptronAppDlg::CMultipayerPerceptronAppDlg(CWnd* pParent /*=NULL*/)
	: CDialog(IDD_MULTIPAYERPERCEPTRONAPP_DIALOG, pParent)
{
	m_hIcon = AfxGetApp()->LoadIcon(IDR_MAINFRAME);
}

void CMultipayerPerceptronAppDlg::DoDataExchange(CDataExchange* pDX)
{
	CDialog::DoDataExchange(pDX);
	DDX_Control(pDX, IDC_GRID_INPUT, m_grdInput);
	DDX_Control(pDX, IDC_GRID_OUTPUT, m_grdOutput);
}

BEGIN_MESSAGE_MAP(CMultipayerPerceptronAppDlg, CDialog)
	ON_WM_SYSCOMMAND()
	ON_BN_CLICKED(IDC_BTN_INPUT, &CMultipayerPerceptronAppDlg::OnBnClickedBtnInput)
	ON_BN_CLICKED(IDC_BTN_RESET, &CMultipayerPerceptronAppDlg::OnBnClickedBtnReset)
	ON_WM_CLOSE()
END_MESSAGE_MAP()


// CMultipayerPerceptronAppDlg message handlers

BOOL CMultipayerPerceptronAppDlg::OnInitDialog()
{
	CDialog::OnInitDialog();

	// Add "About..." menu item to system menu.

	// IDM_ABOUTBOX must be in the system command range.
	ASSERT((IDM_ABOUTBOX & 0xFFF0) == IDM_ABOUTBOX);
	ASSERT(IDM_ABOUTBOX < 0xF000);

	CMenu* pSysMenu = GetSystemMenu(FALSE);
	if (pSysMenu != NULL)
	{
		BOOL bNameValid;
		CString strAboutMenu;
		bNameValid = strAboutMenu.LoadString(IDS_ABOUTBOX);
		ASSERT(bNameValid);
		if (!strAboutMenu.IsEmpty())
		{
			pSysMenu->AppendMenu(MF_SEPARATOR);
			pSysMenu->AppendMenu(MF_STRING, IDM_ABOUTBOX, strAboutMenu);
		}
	}

	// Set the icon for this dialog.  The framework does this automatically
	//  when the application's main window is not a dialog
	SetIcon(m_hIcon, TRUE);			// Set big icon
	SetIcon(m_hIcon, FALSE);		// Set small icon

	// TODO: Add extra initialization here

	// Step 1. grid 생성
	InitGrid();

	// Step 2. 패턴 로드, 실패시 경고 후 리턴
	m_pNetwork = new CMultiPayerPerceptronNetwork();
	if(!m_pNetwork->InitializeNetwork(PATTERN_FILE_NAME))
	{
		AfxMessageBox(L"패턴 로드 실패!", MB_ICONSTOP | MB_OK);
		exit(1);
	}

	// Step 3. 패턴 훈련
	INT nEpoch = m_pNetwork->Train();
	CString strEpoch;
	strEpoch.Format(_T("EPOCH : %d\n"), nEpoch);
	SetDlgItemText(IDC_EPOCH_COUNT, strEpoch);
	
	return TRUE;  // return TRUE  unless you set the focus to a control
}

void CMultipayerPerceptronAppDlg::OnSysCommand(UINT nID, LPARAM lParam)
{
	if ((nID & 0xFFF0) == IDM_ABOUTBOX)
	{
		CAboutDlg dlgAbout;
		dlgAbout.DoModal();
	}
	else
	{
		CDialog::OnSysCommand(nID, lParam);
	}
}

// If you add a minimize button to your dialog, you will need the code below
//  to draw the icon.  For MFC applications using the document/view model,
//  this is automatically done for you by the framework.

void CMultipayerPerceptronAppDlg::OnPaint()
{
	if (IsIconic())
	{
		CPaintDC dc(this); // device context for painting

		SendMessage(WM_ICONERASEBKGND, reinterpret_cast<WPARAM>(dc.GetSafeHdc()), 0);

		// Center icon in client rectangle
		int cxIcon = GetSystemMetrics(SM_CXICON);
		int cyIcon = GetSystemMetrics(SM_CYICON);
		CRect rect;
		GetClientRect(&rect);
		int x = (rect.Width() - cxIcon + 1) / 2;
		int y = (rect.Height() - cyIcon + 1) / 2;

		// Draw the icon
		dc.DrawIcon(x, y, m_hIcon);
	}
	else
	{
		CDialog::OnPaint();
	}
}

// The system calls this function to obtain the cursor to display while the user drags
//  the minimized window.
HCURSOR CMultipayerPerceptronAppDlg::OnQueryDragIcon()
{
	return static_cast<HCURSOR>(m_hIcon);
}

VOID CMultipayerPerceptronAppDlg::OnGridClick(INT a_nRow, INT a_nCol)
{
	INT nBit = m_pNetwork->ToggleInputByGrid(a_nRow, a_nCol);
	if(nBit == 1)
		m_grdInput.SetItemBkColour(a_nRow, a_nCol, RGB(0, 0, 0));
	else
		m_grdInput.SetItemBkColour(a_nRow, a_nCol, RGB(255, 255, 255));

	m_grdInput.Invalidate();
}

VOID CMultipayerPerceptronAppDlg::InitGrid()
{
	m_grdInput.SetEditable(FALSE);
	m_grdInput.EnableSelection(FALSE);
	m_grdInput.EnableScrollBar(SB_BOTH, FALSE);
	m_grdInput.SetEventHandler(this);
	m_grdOutput.SetEditable(FALSE);
	m_grdOutput.EnableSelection(FALSE);
	m_grdOutput.EnableScrollBar(SB_BOTH, FALSE);
	m_grdOutput.SetEventHandler(this);

	// 행/열 갯수 설정
	m_grdInput.SetRowCount(ROW_COL);
	m_grdInput.SetColumnCount(ROW_COL);
	m_grdOutput.SetRowCount(ROW_COL);
	m_grdOutput.SetColumnCount(ROW_COL);

	// 열의 넓이, 행 높이 동시 설정
	INT nCnt = ROW_COL;		// 전체크기 / 원소크기 = 원소개수
	INT nWidthHeight[ROW_COL];

	for(INT i = 0; i < nCnt; i++)
		nWidthHeight[i] = 30;

	for(INT c = 0; c < nCnt; c++)
	{
		for(INT r = 0; r < nCnt; r++)
		{
			m_grdInput.SetColumnWidth(c, nWidthHeight[c]);
			m_grdInput.SetRowHeight(r, nWidthHeight[c]);
			m_grdInput.SetItemFormat(r, c, DT_CENTER);
			m_grdOutput.SetColumnWidth(c, nWidthHeight[c]);
			m_grdOutput.SetRowHeight(r, nWidthHeight[c]);
			m_grdOutput.SetItemFormat(r, c, DT_CENTER);
		}
	}

	m_grdInput.Invalidate();
	m_grdOutput.Invalidate();
}

void CMultipayerPerceptronAppDlg::OnBnClickedBtnInput()
{
	// TODO: Add your control notification handler code here
	INT nResult = m_pNetwork->Run();

	if(nResult < 0)
	{
		CString strEpoch;
		strEpoch.Format(_T("패턴 분류에 실패하였습니다."));
		SetDlgItemText(IDC_SUCCESS_MSG, strEpoch);
		return;
	}

	for(INT c = 0; c < 3; c++)
	{
		for(INT r = 0; r < 3; r++)
		{
			INT nBit = m_pNetwork->getOutputByGrid(r, c);
			if(nBit == 1)
				m_grdOutput.SetItemBkColour(r, c, RGB(0, 0, 0));
			else
				m_grdOutput.SetItemBkColour(r, c, RGB(255, 255, 255));
		}
	}
	CString strEpoch;
	strEpoch.Format(_T("패턴 분류에 성공하였습니다."));
	SetDlgItemText(IDC_SUCCESS_MSG, strEpoch);
	

	m_grdOutput.Invalidate();
}


void CMultipayerPerceptronAppDlg::OnBnClickedBtnReset()
{
	// TODO: Add your control notification handler code here
	m_pNetwork->resetInput();
	for(INT c = 0; c < ROW_COL; c++)
	{
		for(INT r = 0; r < ROW_COL; r++)
		{
			m_grdInput.SetItemBkColour(r, c, RGB(255, 255, 255));
			m_grdOutput.SetItemBkColour(r, c, RGB(255, 255, 255));
		}
	}
	m_grdInput.Invalidate();
	m_grdOutput.Invalidate();
	CString strEpoch;
	strEpoch.Format(_T(""));
	SetDlgItemText(IDC_SUCCESS_MSG, strEpoch);
}


void CMultipayerPerceptronAppDlg::OnClose()
{
	// TODO: Add your message handler code here and/or call default

	// Memory deallocate
	if(m_pNetwork)
		delete m_pNetwork;
	m_pNetwork = NULL;
	

	__super::OnClose();
}


#include "stdafx.h"
#include "MultiPayerPerceptronNetwork.h"

CMultiPayerPerceptronNetwork::CMultiPayerPerceptronNetwork()
{
	m_ppPatterns = NULL;
	m_pInputPattern = NULL;
	m_pOutputPattern = NULL;
	m_unTeachingNode = 0;
	m_nOutputNodeCount = 0;
	m_nPatternAryLength = 0;
	m_pHiddenNodeCounts = NULL;
	m_nHiddenNodeCount = HIDDEN_NODE_COUNT;
	m_nHiddenLayerCount = HIDDEN_LAYER_COUNT;
	m_pHiddenLayers = NULL;
	m_pOutputLayer = NULL;
}

CMultiPayerPerceptronNetwork::~CMultiPayerPerceptronNetwork()
{
	ResetNetwork();
}

INT CMultiPayerPerceptronNetwork::Run(const CHAR* a_pszLearningFile)
{
	if(!ReadPatternFile(a_pszLearningFile))
		return -1;

	return Run();
}

INT CMultiPayerPerceptronNetwork::Run()
{
	PropagateForward(m_pInputPattern);

	for(int i = 0; i < m_nOutputNodeCount; i++)
	{
		// output이 0.8 이상이면 연상 성공인 것으로 한다.
		if(m_pOutputLayer->getNodeOutputVal(i) > DOUBLE(0.8))
		{
			// 출력용 노드에 담는다.
			for(int j = 0; j < m_nPatternAryLength; j++)
				m_pOutputPattern[j] = m_ppPatterns[i][j];

			return i;
		}
			
	}

	// 패턴 분류에 실패했다면 -1 리턴
	return -1;
}

INT CMultiPayerPerceptronNetwork::Train()
{
	// Step 1. 패턴이 학습되었는지 확인
	if(!m_ppPatterns)
		return -1;

	// Step 2. 신경망을 초기화한다.
	if(!InitializeNetwork())
		return -1;

	// Step 3. 히든 레이어 설정한다.
	setNeuranLayer();

	// Step 4. 평가 기준에 부합되거나 최대 반복횟수까지 반복한다.
	int nEpoch = 0;
	for(nEpoch = 0; nEpoch < MAXIMUM_EPOCH; nEpoch++)
	{
		for(int i = 0; i < m_nOutputNodeCount; i++)
		{
			// 순전파
			ASSERT(PropagateForward(m_ppPatterns[i]));

			// 역전파
			ASSERT(PropagateBackward(i));
		}

		if(Evaluate())
			break;
	}

	// 학습이 없다면 -2 리턴
	
	return nEpoch;
}

BOOL CMultiPayerPerceptronNetwork::InitializeNetwork(const CHAR* a_pszFileName)
{
	if(!ReadPatternFile(a_pszFileName))
		return FALSE;

	return InitializeNetwork();
}

BOOL CMultiPayerPerceptronNetwork::InitializeNetwork()
{
//	ResetNetwork();
	if(!m_ppPatterns)
		return FALSE;

	m_pInputPattern = new UINT32[m_nPatternAryLength];
	ZeroMemory(m_pInputPattern, m_nPatternAryLength << 2);
	m_pOutputPattern = new UINT32[m_nPatternAryLength];
	ZeroMemory(m_pOutputPattern, m_nPatternAryLength << 2);

	return TRUE;
}

BOOL CMultiPayerPerceptronNetwork::ReadPatternFile(const CHAR* a_pszFileName)
{
	if(m_ppPatterns != NULL || m_pInputPattern != NULL || m_pOutputPattern != NULL)
		return FALSE;

	FILE *pFile = NULL;
	fopen_s(&pFile, a_pszFileName, "r");
	if(pFile == NULL)
		return FALSE;

	CHAR* pszBuffer = new CHAR[READ_BUFFER_SIZE];
	ZeroMemory(pszBuffer, READ_BUFFER_SIZE);

	// Step 1. 입력 노드 수와 길이가 몇인지 조사한다.
	INT nInputPatternCount = 0;
	INT nInputPatternAryLength = 0;
	INT nInputPatternLength = 0;
	while(fgets(pszBuffer, READ_BUFFER_SIZE, pFile) != NULL)
	{
		INT nCurrPatternLength = 0;
		for(UINT i = 0; i < strlen(pszBuffer); i++)
		{
			switch(pszBuffer[i])
			{
			case '0':
			case '1':
				nCurrPatternLength++;
				break;
			default:
				break;
			}

		}
		nInputPatternCount++;

		if(nCurrPatternLength == 0)
			break;

		// 혹시라도 노드 수가 다르면 FAIL
		if(nInputPatternLength != 0 && nInputPatternLength != nCurrPatternLength)
			return FALSE;

		nInputPatternLength = nCurrPatternLength;
	}
	// 패턴 노드 수 / 32비트
	nInputPatternAryLength = nInputPatternLength >> 5;
	if(nInputPatternLength & 0x1f != 0)
		nInputPatternAryLength++;

	// Step 2. 입력 노드 수를 충족할 만한 동적배열을 할당한다.
	m_ppPatterns = new UINT32*[nInputPatternCount];
	for(int i = 0; i < nInputPatternCount; i++)
	{
		m_ppPatterns[i] = new UINT32[nInputPatternAryLength];
		ZeroMemory(m_ppPatterns[i], nInputPatternAryLength << 2);
	}

	// Step 3. 파일 포인터를 초기화 한 후 파일을 읽어온다.
	ZeroMemory(pszBuffer, READ_BUFFER_SIZE);
	fseek(pFile, 0L, SEEK_SET);
	INT nReadCount = 0;
	INT nMainI = 0;
	INT nSubI = 0;
	while(fgets(pszBuffer, READ_BUFFER_SIZE, pFile) != NULL)
	{
		nSubI = 0;
		for(int i = 0; i < strlen(pszBuffer); i++)
		{
			switch(pszBuffer[i])
			{
			case '0':
				nReadCount++;
				break;
			case '1':
				// 맨 앞자리에 1 삽입 후 시프트
				m_ppPatterns[nMainI][nSubI] |= (0x80000000 >> nReadCount);
				nReadCount++;
				break;
			default:
				break;
			}

			if(nReadCount >= 0x20)
			{
				nSubI++;
				nReadCount = 0;
			}
		}

		nMainI++;
		ZeroMemory(pszBuffer, READ_BUFFER_SIZE);
		nReadCount = 0;
	}

	fclose(pFile);

	// 멤버변수 설정
	m_nOutputNodeCount = nInputPatternCount;
	m_nPatternLength = nInputPatternLength;
	m_nPatternAryLength = nInputPatternAryLength;

	delete pszBuffer;

	return TRUE;
}

VOID CMultiPayerPerceptronNetwork::setNeuranLayer()
{
	// 은닉층 설정
	if(m_pHiddenLayers)
		delete m_pHiddenLayers;

	m_pHiddenLayers = new CNeuranLayer[m_nHiddenLayerCount];

	// HiddenLayerNode 수가 서로 같을 경우
	INT nPrevNodeCount = m_nPatternLength;
	if(m_nHiddenNodeCount != 0)
	{
		for(int i = 0; i < m_nHiddenLayerCount; i++)
		{
			m_pHiddenLayers[i].makeNodeLayer(m_nHiddenNodeCount, nPrevNodeCount);
			nPrevNodeCount = m_nHiddenNodeCount;
		}
	}
	else
	{
		for(int i = 0; i < m_nHiddenLayerCount; i++)
		{
			m_pHiddenLayers[i].makeNodeLayer(m_pHiddenNodeCounts[i], nPrevNodeCount);
			nPrevNodeCount = m_nHiddenNodeCount;
		}
	}

	// 출력층 설정
	if(m_pOutputLayer)
		delete m_pOutputLayer;

	m_pOutputLayer = new CNeuranLayer(m_nOutputNodeCount, nPrevNodeCount);
}

VOID CMultiPayerPerceptronNetwork::setNeuranLayer(INT a_nLayerCount, INT a_nNodeCount)
{
	m_nHiddenLayerCount = a_nLayerCount;
	m_nHiddenNodeCount = a_nNodeCount;
	if(m_pHiddenNodeCounts)
		delete m_pHiddenNodeCounts;
	m_pHiddenNodeCounts = NULL;

	setNeuranLayer();
}

VOID CMultiPayerPerceptronNetwork::setNeuranLayer(INT a_nLayerCount, INT* a_pNodeCountArray)
{
	m_nHiddenLayerCount = a_nLayerCount;
	m_pHiddenNodeCounts = new INT[a_nLayerCount];

	try
	{
		for(int i = 0; i < a_nLayerCount; i++)
			m_pHiddenNodeCounts[i] = a_pNodeCountArray[i];
	}
	catch(int e)
	{
		delete m_pHiddenNodeCounts;
		m_pHiddenNodeCounts = NULL;
		return;
	}

	m_nHiddenNodeCount = 0;

	setNeuranLayer();
}

BOOL CMultiPayerPerceptronNetwork::PropagateForward(const UINT32* a_pPattern)
{
	if(!m_ppPatterns || !m_pHiddenLayers)
		return FALSE;

	const UINT32* pInputPattern = a_pPattern;

	// Step 1. 입력층 -> 은닉층 계산 (힌트: 은닉층 관점에서 계산한다고 생각하면 편함)
	INT nNodeCount = m_pHiddenLayers[0].getNodeCount();
	CNeuranNode* pNode = NULL;
	for(int i = 0; i < nNodeCount; i++)
	{
		pNode = m_pHiddenLayers[0].getNode(i);

		DOUBLE dInputValue = 0;
		for(int j = 0; j < m_nPatternLength; j++)
		{
			INT nMainI = j >> 5;
			INT nSubI = j & 0x1f;
			UINT32 nBlock = 0x80000000 >> nSubI;
			INT nBit = !!(pInputPattern[nMainI] & nBlock);
			dInputValue += m_pHiddenLayers[0].getNode(i)->getWeight(j) * DOUBLE(nBit);
		}

		pNode->setInputVal(dInputValue);
	}

	// Step 2. (은닉층이 2개 이상 존재 시) 은닉층 -> 은닉층 계산
	CNeuranLayer* pPrevLayer = NULL;
	if(m_nHiddenLayerCount > 1)
	{
		for(int nLevel = 1; nLevel < m_nHiddenLayerCount; nLevel++)
		{
			pPrevLayer = &(m_pHiddenLayers[nLevel - 1]);
			nNodeCount = m_pHiddenLayers[nLevel].getNodeCount();
			for(int i = 0; i < nNodeCount; i++)
			{
				pNode = m_pHiddenLayers[nLevel].getNode(i);

				DOUBLE dInputValue = 0;
				INT nPrevNodeCount = pPrevLayer->getNodeCount();
				for(int j = 0; j < nPrevNodeCount; j++)
					dInputValue += m_pHiddenLayers[nLevel].getNode(i)->getWeight(j) * 
						pPrevLayer->getNodeOutputVal(j);

				pNode->setInputVal(dInputValue);
			}
		}
	}

	// Step 3. 은닉층 -> 출력층 계산
	nNodeCount = m_pOutputLayer->getNodeCount();
	pPrevLayer = &(m_pHiddenLayers[m_nHiddenLayerCount - 1]);
	for(int i = 0; i < nNodeCount; i++)
	{
		pNode = m_pOutputLayer->getNode(i);

		DOUBLE dInputValue = 0;
		INT nPrevNodeCount = pPrevLayer->getNodeCount();
		for(int j = 0; j < nPrevNodeCount; j++)
			dInputValue += m_pOutputLayer->getNode(i)->getWeight(j) *
				pPrevLayer->getNodeOutputVal(j);

		pNode->setInputVal(dInputValue);
	}

	return TRUE;
}

INT CMultiPayerPerceptronNetwork::PropagateBackward(INT a_nPatternIndex)
{
	if(!m_ppPatterns)
		return -1;

	const UINT32 unTeachingBit = 0x80000000 >> a_nPatternIndex;
	const UINT32* pInputPattern = m_ppPatterns[a_nPatternIndex];

	// Step 1. 은닉층 -> 출력층 델타 값 계산
	m_pOutputLayer->setOutputDelta(unTeachingBit, m_nOutputNodeCount);

	// Step 2. 은닉층 델타 값 계산 
	CNeuranLayer* pUpperLayer = m_pOutputLayer;
	for(int nLevel = m_nHiddenLayerCount - 1; nLevel >= 0; nLevel--)
	{
		m_pHiddenLayers[nLevel].setHiddenDelta(pUpperLayer);
		pUpperLayer = &(m_pHiddenLayers[nLevel]);
	}

	// Step 3. 출력층 가중치 수정
	m_pOutputLayer->RecalibrateWeight(&(m_pHiddenLayers[m_nHiddenLayerCount-1]));

	// Step 4. 은닉층 가중치 수정
	for(int nLevel = m_nHiddenLayerCount - 1; nLevel > 0; nLevel--)
		m_pHiddenLayers[nLevel].RecalibrateWeight(&(m_pHiddenLayers[nLevel - 1]));
	m_pHiddenLayers[0].RecalibrateWeight(pInputPattern, m_nPatternLength);

	return TRUE;
}

BOOL CMultiPayerPerceptronNetwork::Evaluate()
{
	DOUBLE dMaxErrorDiffValue = 0;

	for(int i = 0; i < m_nHiddenLayerCount; i++)
	{
		if(dMaxErrorDiffValue < m_pHiddenLayers[i].getMaxWeightDiff())
			dMaxErrorDiffValue = m_pHiddenLayers[i].getMaxWeightDiff();
	}
	if(dMaxErrorDiffValue < m_pOutputLayer->getMaxWeightDiff())
		dMaxErrorDiffValue = m_pOutputLayer->getMaxWeightDiff();
	
	if(dMaxErrorDiffValue > ERROR_ALLOWED_VAL)
		return FALSE;
	return TRUE;
}

VOID CMultiPayerPerceptronNetwork::ResetNetwork()
{
	m_unTeachingNode = 0;
	m_nOutputNodeCount = 0;
	m_nPatternAryLength = 0;
	m_nHiddenNodeCount = HIDDEN_NODE_COUNT;
	m_nHiddenLayerCount = HIDDEN_LAYER_COUNT;

	if(m_ppPatterns)
	{
		for(int i = 0; i < m_nPatternAryLength; i++)
			if(m_ppPatterns[i])
				delete[] m_ppPatterns[i];
		delete[] m_ppPatterns;
	}
	m_ppPatterns = NULL;

	if(!m_pInputPattern)
		delete[] m_pInputPattern;
	m_pInputPattern = NULL;

	if(!m_pOutputPattern)
		delete[] m_pOutputPattern;
	m_pOutputPattern = NULL;

	if(m_pOutputLayer)
		delete m_pOutputLayer;
	m_pOutputLayer = NULL;

	if(m_pHiddenLayers)
		delete[] m_pHiddenLayers;
	m_pHiddenLayers = NULL;

	
}

// 정사각형으로 되어있다고 가정
INT CMultiPayerPerceptronNetwork::ToggleInputByGrid(INT a_nRow, INT a_nCol)
{
	if(!m_pInputPattern)
		return -1;

	INT nSquareLength = (INT)sqrt(m_nPatternLength);
	INT nIndex = a_nRow * nSquareLength + a_nCol;
	INT nMainI = nIndex >> 5;
	INT nSubI = nIndex & 0x1f;
	UINT32 nBlock = 0x80000000 >> nSubI;
	m_pInputPattern[nMainI] = ((m_pInputPattern[nMainI] ^ nBlock) & nBlock) | (m_pInputPattern[nMainI] & ~nBlock);
	
	return !!(m_pInputPattern[nMainI] & nBlock);
}

INT CMultiPayerPerceptronNetwork::getInputByGrid(INT a_nRow, INT a_nCol)
{
	if(!m_pInputPattern)
		return -1;

	INT nSquareLength = (INT)sqrt(m_nPatternLength);
	INT nIndex = a_nRow * nSquareLength + a_nCol;
	INT nMainI = nIndex >> 5;
	INT nSubI = nIndex & 0x1f;
	UINT32 nBlock = 0x80000000 >> nSubI;

	return !!(m_pInputPattern[nMainI] & nBlock);
}

INT CMultiPayerPerceptronNetwork::getOutputByGrid(INT a_nRow, INT a_nCol)
{
	if(!m_pOutputPattern)
		return -1;

	INT nSquareLength = (INT)sqrt(m_nPatternLength);
	INT nIndex = a_nRow * nSquareLength + a_nCol;
	INT nMainI = nIndex >> 5;
	INT nSubI = nIndex & 0x1f;
	UINT32 nBlock = 0x80000000 >> nSubI;

	return !!(m_pOutputPattern[nMainI] & nBlock);
}
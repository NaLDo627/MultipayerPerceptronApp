#include "stdafx.h"
#include "MultiPayerPerceptronNetwork.h"

MultiPayerPerceptronNetwork::MultiPayerPerceptronNetwork()
{
	m_ppPatterns = NULL;
	m_pInputPattern = NULL;
	m_pOutputPattern = NULL;
	m_unOutputNodes = 0;
	m_aHiddenLayerNodes = NULL;
	m_aHiddenOutputWeight = NULL;
	m_aInputHiddenWeight = NULL;
	m_nTeachingNodeCount = 0;
	m_nPatternAryCount = 0;
}

MultiPayerPerceptronNetwork::~MultiPayerPerceptronNetwork()
{
	if(m_ppPatterns)
	{
		for(int i = 0; i < m_nPatternAryCount; i++)
			if(m_ppPatterns[i])
				delete m_ppPatterns[i];
		delete m_ppPatterns;
	}
}

BOOL MultiPayerPerceptronNetwork::Run(const CHAR* a_pszLearningFile, const UINT32* a_pInputPattern, UINT32* a_pOutputPattern)
{

	return TRUE;
}


BOOL MultiPayerPerceptronNetwork::ReadPatternFile(const CHAR* a_pszFileName)
{
	FILE *pFile = NULL;
	fopen_s(&pFile, a_pszFileName, "r");
	if(pFile == NULL)
		return FALSE;

	CHAR* pszBuffer = new CHAR[READ_BUFFER_SIZE];
	ZeroMemory(pszBuffer, READ_BUFFER_SIZE);

	// Step 1. 입력 노드 수와 길이가 몇인지 조사한다.
	INT nInputPatternCount = 0;
	INT nInputPatternAryCount = 0;
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
	nInputPatternAryCount = nInputPatternLength / (sizeof(UINT32) * 8);
	if(nInputPatternLength % (sizeof(UINT32) * 8) != 0)
		nInputPatternAryCount++;

	// Step 2. 입력 노드 수를 충족할 만한 동적배열을 할당한다.
	m_ppPatterns = new UINT32*[nInputPatternCount];
	for(int i = 0; i < nInputPatternCount; i++)
	{
		m_ppPatterns[i] = new UINT32[nInputPatternAryCount];
		ZeroMemory(m_ppPatterns[i], sizeof(UINT32) * nInputPatternAryCount);
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
				// m_ppPatterns[nMainI][nSubI] = m_ppPatterns[nMainI][nSubI] >> 1;
				nReadCount++;
				break;
			case '1':
				// 맨 앞자리에 1 삽입 후 시프트
				m_ppPatterns[nMainI][nSubI] |= (0x80000000 >> nReadCount);
				// m_ppPatterns[nMainI][nSubI] = m_ppPatterns[nMainI][nSubI] >> 1;
				nReadCount++;
				break;
			default:
				break;
			}

			if(nReadCount >= (sizeof(UINT32) * 8))
			{
				nSubI++;
				nReadCount = 0;
			}
		}

		nMainI++;
		ZeroMemory(pszBuffer, READ_BUFFER_SIZE);
	}

	fclose(pFile);

	m_nTeachingNodeCount = nInputPatternCount;
	m_nPatternAryCount = nInputPatternAryCount;

	delete pszBuffer;

	return TRUE;
}

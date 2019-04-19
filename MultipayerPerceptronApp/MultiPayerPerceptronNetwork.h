#pragma once
#include "resource.h"		// main symbols
#include "GridCtrl\GridCtrl.h"

#define READ_BUFFER_SIZE	1024
//#define HIDDEN_NODE_COUNT	32
//#define HIDDEN_LAYER_COUNT	2
#define INITIAL_WEIGHT		1
#define INITIAL_ETA			(0.1)
#define MAXIMUM_EPOCH		400000
#define ERROR_ALLOWED_VAL	0.002

#define HIDDEN_NODE_COUNT	2
#define HIDDEN_LAYER_COUNT	1


/* 뉴런의 노드 */
class CNeuranNode
{
public:
	VOID setInputVal(DOUBLE a_dValue)
	{ 
		m_dInputValue = a_dValue;
		m_dOutputValue = SigmoidFunction(a_dValue);
	}

	DOUBLE getOutputVal(){ return m_dOutputValue; }

public:
	CNeuranNode(){ m_dInputValue = 0; m_dOutputValue = 0; }
	CNeuranNode(DOUBLE a_dValue){setInputVal(a_dValue); }
	~CNeuranNode(){ }

private:
	DOUBLE SigmoidFunction(DOUBLE a_dInput){ return (1 / (1 + exp(a_dInput * DOUBLE(-1)))); }

private:
	DOUBLE m_dInputValue;
	DOUBLE m_dOutputValue;
};


/* 뉴런의 레이어 (노드의 집합) */
class CNeuranLayer 
{ // 망 클래스에서 입력 레이어는 이 클래스를 사용하지 않는 걸로 함 (히든, 출력 레이어만 사용)
public:
	VOID makeNodeLayer(INT a_nNodeCount, INT a_nPrevNodeCount)
	{
		if(m_pNodes)
			delete m_pNodes;

		// 내부 노드 생성
		m_pNodes = new CNeuranNode[a_nNodeCount];
		m_ppWeights = new DOUBLE*[a_nPrevNodeCount];
		for(int i = 0; i < a_nPrevNodeCount; i++)
			m_ppWeights[i] = new DOUBLE[a_nNodeCount];

		// 노드 간 가중치 생성
		for(int i = 0; i < a_nPrevNodeCount; i++)
			for(int j = 0; j < a_nNodeCount; j++)
				m_ppWeights[i][j] = INITIAL_WEIGHT;

		m_nNodeCount = a_nNodeCount;
		m_nPrevNodeCount = a_nPrevNodeCount;
	}

	DOUBLE getNodeOutputVal(INT a_nIndex)
	{
		if(!m_pNodes) return -1;
		if(a_nIndex < 0 || a_nIndex >= m_nNodeCount) return -1;
		return m_pNodes[a_nIndex].getOutputVal();
	}

	VOID InputUnit(INT a_nIndex, DOUBLE a_dValue)
	{
		if(!m_pNodes) return;
		if(a_nIndex < 0 || a_nIndex >= m_nNodeCount) return;
		m_pNodes[a_nIndex].setInputVal(a_dValue);
	}

	VOID setOutputDelta(const UINT32 a_unOutput, INT a_nOutputLength)
	{
		if(a_nOutputLength != m_nNodeCount)
			return;

		if(!m_pDeltas)
			m_pDeltas = new DOUBLE[m_nNodeCount];

		for(int i = 0; i < m_nNodeCount; i++)
		{
			UINT32 unBlock = 0x80000000 >> i;
			INT nBit = !!(a_unOutput & unBlock);

			m_pDeltas[i] = m_pNodes[i].getOutputVal() * (1 - m_pNodes[i].getOutputVal()) * (DOUBLE(nBit) - m_pNodes[i].getOutputVal());
		}
	}

	VOID setHiddenDelta(CNeuranLayer* a_pUpperLayer)
	{
		if(!m_pDeltas)
			m_pDeltas = new DOUBLE[m_nNodeCount];

		for(int i = 0; i < m_nNodeCount; i++)
		{
			INT nUpperNodeCount = a_pUpperLayer->getNodeCount();
			DOUBLE dError = 0;
			for(int j = 0; j < nUpperNodeCount; j++)
				dError += a_pUpperLayer->getDelta(j) * a_pUpperLayer->getWeight(i, j);
			m_pDeltas[i] = m_pNodes[i].getOutputVal() * (1 - m_pNodes[i].getOutputVal()) * (dError);
		}
	}

	DOUBLE getDelta(INT a_nIndex) { return m_pDeltas[a_nIndex]; }
	CNeuranNode* getNode(INT a_nIndex) { return &(m_pNodes[a_nIndex]); }
	INT getNodeCount() { return m_nNodeCount; }

	DOUBLE getWeight(INT a_nPrevI, INT a_nCurrI) { return m_ppWeights[a_nPrevI][a_nCurrI]; }
	DOUBLE getMaxWeightDiff() { return m_dMaxWeightDiff; }
	VOID setWeight(INT a_nPrevI, INT a_nCurrI, DOUBLE a_dWeight) { m_ppWeights[a_nPrevI][a_nCurrI] = a_dWeight; }
	VOID RecalibrateWeight(CNeuranLayer* a_pLowerLayer)
	{
		if(!m_pDeltas || !a_pLowerLayer)
			return;

		m_dMaxWeightDiff = 0;

		for(int i = 0; i < a_pLowerLayer->getNodeCount(); i++)
			for(int j = 0; j < m_nNodeCount; j++)
			{
				DOUBLE dOldWeight = m_ppWeights[i][j];
				m_ppWeights[i][j] = dOldWeight + INITIAL_ETA * m_pDeltas[j] * a_pLowerLayer->getNode(i)->getOutputVal();
				if(m_dMaxWeightDiff < fabs(m_ppWeights[i][j] - dOldWeight))
					m_dMaxWeightDiff = fabs(m_ppWeights[i][j] - dOldWeight);
			}
	}

	VOID RecalibrateWeight(const UINT32* a_pInputPattern, INT a_nPatternLength)
	{
		if(!m_pDeltas || !a_pInputPattern)
			return;

		m_dMaxWeightDiff = 0;

		for(int i = 0; i < a_nPatternLength; i++)
		{
			INT nMainI = i >> 5;
			INT nSubI = i & 0x1f;
			UINT32 unBlock = 0x80000000 >> i;
			INT nBits = !!(a_pInputPattern[nMainI] & unBlock);

			for(int j = 0; j < m_nNodeCount; j++)
			{
				DOUBLE dOldWeight = m_ppWeights[i][j];
				m_ppWeights[i][j] = dOldWeight + INITIAL_ETA * m_pDeltas[j] * DOUBLE(nBits);
				if(m_dMaxWeightDiff < fabs(m_ppWeights[i][j] - dOldWeight))
					m_dMaxWeightDiff = fabs(m_ppWeights[i][j] - dOldWeight);
			}
		}
	}

public:
	CNeuranLayer()
	{
		m_pNodes = NULL;
		m_ppWeights = NULL;
		m_pDeltas = NULL;
		m_nNodeCount = 0;
		m_nPrevNodeCount = 0;
		m_dMaxWeightDiff = 0;
	}

	CNeuranLayer(INT a_nNodeCount, INT a_nPrevNodeCount)
	{
		m_pNodes = NULL;
		m_ppWeights = NULL;
		m_pDeltas = NULL;
		m_nNodeCount = 0;
		m_nPrevNodeCount = 0;
		m_dMaxWeightDiff = 0;
		makeNodeLayer(a_nNodeCount, a_nPrevNodeCount);
	}

	~CNeuranLayer()
	{
		if(m_pNodes)
			delete m_pNodes;

		if(m_pDeltas)
			delete m_pDeltas;

		if(m_ppWeights)
		{
			for(int i = 0; i < m_nPrevNodeCount; i++)
				delete m_ppWeights[i];

			delete m_ppWeights;
		}
	}

private:
	CNeuranNode* m_pNodes;
	DOUBLE**	 m_ppWeights; // 현재 레이어로 들어오는 가중치들의 이중 배열 (1차원 : 이전 노드 좌표, 2차원 : 현재 노드 좌표)
	DOUBLE*		 m_pDeltas;
	DOUBLE		 m_dMaxWeightDiff;
	INT			 m_nNodeCount;
	INT			 m_nPrevNodeCount;
};

/* 다중 퍼셉트론 네트워크 */
class CMultiPayerPerceptronNetwork
{
public:
	// Common
	INT Run(const CHAR* a_pszLearningFile, const UINT32* a_pInputPattern, UINT32* a_pOutputPattern);
	INT Run(const UINT32* a_pInputPattern, UINT32* a_pOutputPattern);
	INT Train();

	// Neuran
	BOOL	ReadPatternFile(const CHAR* a_pszFileName);
	BOOL	InitializeNetwork(const CHAR* a_pszFileName);
	BOOL	InitializeNetwork();
	VOID	setNeuranLayer();
	VOID	setNeuranLayer(INT a_nLayerCount, INT a_nNodeCount);
	VOID	setNeuranLayer(INT a_nLayerCount, INT* a_pNodeCountArray);
	
	BOOL	PropagateForward(INT a_nPatternIndex);
	BOOL	PropagateBackward(INT a_nPatternIndex);
	BOOL	Evaluate();

	VOID	ResetNetwork();

	// Matrix
	INT		ToggleInputByGrid(INT a_nRow, INT a_nCol);
	INT		getInputByGrid(INT a_nRow, INT a_nCol);

public:
	CMultiPayerPerceptronNetwork();
	~CMultiPayerPerceptronNetwork();

private:
	// Hidden Layer - ary
	CNeuranLayer* m_pHiddenLayers;

	// Output Layer
	CNeuranLayer* m_pOutputLayer;

	// Patterns
	UINT32**	m_ppPatterns;
	UINT32*		m_pInputPattern;
	UINT32*		m_pOutputPattern;
	UINT32		m_unTeachingNode;
	
	INT			m_nOutputNodeCount;
	INT			m_nPatternLength;
	INT			m_nPatternAryLength;
	INT			m_nHiddenLayerCount;
	INT			m_nHiddenNodeCount;
	INT*		m_pHiddenNodeCounts;

	// Training
	INT			m_nEpoch;
	BOOL		m_bIsTrained;
};
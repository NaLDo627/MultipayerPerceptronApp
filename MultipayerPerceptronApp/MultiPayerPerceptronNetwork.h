#pragma once
#include "resource.h"		// main symbols
#include "GridCtrl\GridCtrl.h"

#define READ_BUFFER_SIZE	1024
#define NODE_COUNT			121
#define HIDDEN_NODE_COUNT	32
#define HIDDEN_LAYER_COUNT	2
#define INITIAL_WEIGHT		1
#define INITIAL_ETA			(0.1)
#define MAXIMUM_EPOCH		400000


/* 뉴런의 노드 */
class CNeuranNode
{
public:
	DOUBLE setInputVal(DOUBLE a_dValue)
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
	DOUBLE SigmoidFunction(DOUBLE a_dInput){ return (1 / exp(a_dInput * DOUBLE(-1))); }

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
		m_nPrevNodeCount = 0;
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

	CNeuranNode* getNode(INT a_nIndex) { return &(m_pNodes[a_nIndex]); }
	INT getNodeCount() { return m_nNodeCount; }
	DOUBLE getWeight(INT a_nPrevI, INT a_nCurrI) { return m_ppWeights[a_nPrevI][a_nCurrI]; }
	VOID setWeight(INT a_nPrevI, INT a_nCurrI, DOUBLE a_dWeight) { m_ppWeights[a_nPrevI][a_nCurrI] = a_dWeight; }

public:
	CNeuranLayer()
	{
		m_pNodes = NULL;
		m_ppWeights = NULL;
		m_nNodeCount = 0;
		m_nPrevNodeCount = 0;
	}

	CNeuranLayer(INT a_nNodeCount, INT a_nPrevNodeCount)
	{
		m_pNodes = NULL;
		m_ppWeights = NULL;
		m_nNodeCount = 0;
		m_nPrevNodeCount = 0;
		makeNodeLayer(a_nNodeCount, a_nPrevNodeCount);
	}

	~CNeuranLayer()
	{
		if(m_pNodes)
			delete m_pNodes;

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


	// Neuran
	BOOL	ReadPatternFile(const CHAR* a_pszFileName);
	BOOL	InitializeNetwork(const CHAR* a_pszFileName);
	BOOL	InitializeNetwork();
	VOID	setHiddenLayer();
	VOID	setHiddenLayer(INT a_nLayerCount, INT a_nNodeCount);
	VOID	setHiddenLayer(INT a_nLayerCount, INT* a_pNodeCountArray);
	
	INT		PropagateForward(INT a_nPatternIndex);
	INT		PropagateBackward(INT a_nPatternIndex);
	DOUBLE	CalcluateOutputDelta(DOUBLE a_dOutput, UINT32 a_unTeacherValue);
	DOUBLE	CalcluateHiddenDelta(DOUBLE a_dOutput, DOUBLE* a_aArray);
	BOOL	Evaluate();

	VOID	RecalibrateWeight();
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
};
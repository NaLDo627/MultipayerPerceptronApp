#pragma once
#include "resource.h"		// main symbols
#include "GridCtrl\GridCtrl.h"

#define READ_BUFFER_SIZE	1024
//#define HIDDEN_NODE_COUNT	16
//#define HIDDEN_LAYER_COUNT	1
#define INITIAL_WEIGHT		(0.1)
//#define INITIAL_ETA			(0.56308)
#define INITIAL_ETA			(1)
#define MAXIMUM_EPOCH		50000
#define ERROR_ALLOWED_VAL	0.0002

#define HIDDEN_NODE_COUNT	2
#define HIDDEN_LAYER_COUNT	1


/* 뉴런의 노드 */
class CNeuranNode
{
public:
	VOID InitNode(DOUBLE a_dValue, INT a_nPrevNodeCount)
	{
		m_dInputValue = a_dValue;
		m_dOutputValue = SigmoidFunction(a_dValue);
		m_nPrevNodeCount = a_nPrevNodeCount;
		m_pWeight = new DOUBLE[m_nPrevNodeCount];
		for(int i = 0; i < m_nPrevNodeCount; i++)
			m_pWeight[i] = INITIAL_WEIGHT;
	}

	VOID setInputVal(DOUBLE a_dValue)
	{
		m_dInputValue = a_dValue;
		m_dOutputValue = SigmoidFunction(a_dValue);
	}

	VOID setWeights(INT a_nPrevNodeCount)
	{
		m_pWeight = new DOUBLE[a_nPrevNodeCount];
		for(int i = 0; i < a_nPrevNodeCount; i++)
			m_pWeight[i] = INITIAL_WEIGHT;
	}

	DOUBLE getWeight(INT a_nPrevNodeIndex){ return m_pWeight[a_nPrevNodeIndex]; }
	VOID   setWeight(INT a_nPrevNodeIndex, DOUBLE a_dDouble){ m_pWeight[a_nPrevNodeIndex] = a_dDouble; }
	VOID   setDelta(DOUBLE a_dDelta) { m_dDelta = a_dDelta; }
	DOUBLE getDelta() { return m_dDelta; }

	DOUBLE getOutputVal(){ return m_dOutputValue; }
	VOID freeWeight() {
		if(m_pWeight) delete[] m_pWeight;
		m_pWeight = NULL;
	}

public:
	CNeuranNode()
	{
		m_dInputValue = 0; 
		m_dOutputValue = 0; 
		m_nPrevNodeCount = 0; 
		m_pWeight = NULL;
		m_dDelta = 0;
	}
	//CNeuranNode(DOUBLE a_dValue){setInputVal(a_dValue); }
	~CNeuranNode(){
		freeWeight(); 
	}

private:
	DOUBLE SigmoidFunction(DOUBLE a_dInput){ return (1 / (1 + exp(a_dInput * DOUBLE(-1)))); }

private:
	DOUBLE	m_dInputValue;
	DOUBLE	m_dOutputValue;
	INT		m_nPrevNodeCount;
	DOUBLE* m_pWeight;
	DOUBLE  m_dDelta;
};


/* 뉴런의 레이어 (노드의 집합) */
class CNeuranLayer 
{ // 망 클래스에서 입력 레이어는 이 클래스를 사용하지 않는 걸로 함 (히든, 출력 레이어만 사용)
public:
	VOID makeNodeLayer(INT a_nNodeCount, INT a_nPrevNodeCount)
	{
		if(m_pNodes)
			delete[] m_pNodes;

		// 내부 노드 생성
		m_pNodes = new CNeuranNode[a_nNodeCount];
		for(int i = 0; i < a_nNodeCount; i++)
			m_pNodes[i].setWeights(a_nPrevNodeCount);

		m_nNodeCount = a_nNodeCount;
		m_nPrevNodeCount = a_nPrevNodeCount;
	}

	DOUBLE getNodeOutputVal(INT a_nIndex)
	{
		if(!m_pNodes) 
			return -1;
		if(a_nIndex < 0 || a_nIndex >= m_nNodeCount) 
			return -1;
		return m_pNodes[a_nIndex].getOutputVal();
	}

	VOID InputUnit(INT a_nIndex, DOUBLE a_dValue, INT a_nPrevNodeCount)
	{
		if(!m_pNodes) return;
		if(a_nIndex < 0 || a_nIndex >= m_nNodeCount) return;
		m_pNodes[a_nIndex].InitNode(a_dValue, a_nPrevNodeCount);
	}

	VOID setOutputDelta(const UINT32 a_unOutput, INT a_nOutputLength)
	{
		if(a_nOutputLength != m_nNodeCount)
			return;

		for(int i = 0; i < m_nNodeCount; i++)
		{
			UINT32 unBlock = 0x80000000 >> i;
			INT nBit = !!(a_unOutput & unBlock);
			DOUBLE dDelta = m_pNodes[i].getOutputVal() * (DOUBLE(1) - m_pNodes[i].getOutputVal()) * (DOUBLE(nBit) - m_pNodes[i].getOutputVal());
			m_pNodes[i].setDelta(dDelta);
		}
	}

	VOID setHiddenDelta(CNeuranLayer* a_pUpperLayer)
	{
		for(int i = 0; i < m_nNodeCount; i++)
		{
			INT nUpperNodeCount = a_pUpperLayer->getNodeCount();
			DOUBLE dError = 0;
			for(int j = 0; j < nUpperNodeCount; j++)
				dError += a_pUpperLayer->getNode(j)->getDelta() * a_pUpperLayer->getNode(j)->getWeight(i);
			DOUBLE dDelta = m_pNodes[i].getOutputVal() * (DOUBLE(1) - m_pNodes[i].getOutputVal()) * (dError);
			m_pNodes[i].setDelta(dDelta);
		}
	}

	CNeuranNode* getNode(INT a_nIndex) { return &(m_pNodes[a_nIndex]); }
	INT getNodeCount() { return m_nNodeCount; }

	DOUBLE getMaxWeightDiff() { return m_dMaxWeightDiff; }
	VOID RecalibrateWeight(CNeuranLayer* a_pLowerLayer)
	{
		if(!a_pLowerLayer)
			return;

		m_dMaxWeightDiff = 0;

		for(int i = 0; i < a_pLowerLayer->getNodeCount(); i++)
			for(int j = 0; j < m_nNodeCount; j++)
			{
				DOUBLE dOldWeight = m_pNodes[j].getWeight(i);
				DOUBLE dNewWeight = dOldWeight + INITIAL_ETA * m_pNodes[j].getDelta() * a_pLowerLayer->getNodeOutputVal(i);
				if(m_dMaxWeightDiff < fabs(dNewWeight - dOldWeight))
					m_dMaxWeightDiff = fabs(dNewWeight - dOldWeight);
				m_pNodes[j].setWeight(i, dNewWeight);
			}
	}

	VOID RecalibrateWeight(const UINT32* a_pInputPattern, INT a_nPatternLength)
	{
		if(!a_pInputPattern)
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
				DOUBLE dOldWeight = m_pNodes[j].getWeight(i);
				DOUBLE dNewWeight = dOldWeight + INITIAL_ETA * m_pNodes[j].getDelta() * DOUBLE(nBits);
				if(m_dMaxWeightDiff < fabs(dNewWeight - dOldWeight))
					m_dMaxWeightDiff = fabs(dNewWeight - dOldWeight);
				m_pNodes[j].setWeight(i, dNewWeight);
			}
		}
	}

	VOID freeLayer()
	{
		for(int i = 0; i < m_nNodeCount; i++)
			(&m_pNodes[i])->freeWeight();

		if(m_pNodes)
			delete[] m_pNodes;
		m_pNodes = NULL;
	}

public:
	CNeuranLayer()
	{
		m_pNodes = NULL;
		m_nNodeCount = 0;
		m_nPrevNodeCount = 0;
		m_dMaxWeightDiff = 0;
	}

	CNeuranLayer(INT a_nNodeCount, INT a_nPrevNodeCount)
	{
		m_pNodes = NULL;
		m_nNodeCount = 0;
		m_nPrevNodeCount = 0;
		m_dMaxWeightDiff = 0;
		makeNodeLayer(a_nNodeCount, a_nPrevNodeCount);
	}

	~CNeuranLayer()
	{
		freeLayer();
	}

private:
	CNeuranNode* m_pNodes;
	DOUBLE		 m_dMaxWeightDiff;
	INT			 m_nNodeCount;
	INT			 m_nPrevNodeCount;
};

/* 다중 퍼셉트론 네트워크 */
class CMultiPayerPerceptronNetwork
{
public:
	// Common
	INT Run(const CHAR* a_pszLearningFile);
	INT Run();
	INT Train();
	VOID resetInput() 
	{ 
		ZeroMemory(m_pInputPattern, m_nPatternAryLength << 2);
	}

	// Neuran
	BOOL	ReadPatternFile(const CHAR* a_pszFileName);
	BOOL	InitializeNetwork(const CHAR* a_pszFileName);
	BOOL	InitializeNetwork();
	VOID	setNeuranLayer();
	VOID	setNeuranLayer(INT a_nLayerCount, INT a_nNodeCount);
	VOID	setNeuranLayer(INT a_nLayerCount, INT* a_pNodeCountArray);
	
	BOOL	PropagateForward(const UINT32* a_pPattern);
	BOOL	PropagateBackward(INT a_nPatternIndex);
	BOOL	Evaluate();

	VOID	ResetNetwork();
	VOID	setOutputCount(INT a_nOutputCount) { m_nOutputNodeCount = a_nOutputCount; }

	// Matrix
	INT		ToggleInputByGrid(INT a_nRow, INT a_nCol);
	INT		getInputByGrid(INT a_nRow, INT a_nCol);
	INT		getOutputByGrid(INT a_nRow, INT a_nCol);

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
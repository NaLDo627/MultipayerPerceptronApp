#pragma once
#include "resource.h"		// main symbols
#include "GridCtrl\GridCtrl.h"

#define READ_BUFFER_SIZE	1024
#define NODE_COUNT			121
#define HIDDEN_NEURAN_COUNT	32
#define HIDDEN_LAYER_COUNT	1
#define INITIAL_WEIGHT		1
#define INITIAL_ETC			(0.1)


class MultiPayerPerceptronNetwork
{
public:
	// Common
	BOOL Run(const CHAR* a_pszLearningFile, const UINT32* a_pInputPattern, UINT32* a_pOutputPattern);

	// Neuran
	BOOL   ReadPatternFile(const CHAR* a_pszFileName);
	DOUBLE SigmoidFunction(DOUBLE a_dInput){ return (1 / exp(a_dInput * DOUBLE(-1))); }
	DOUBLE CalcluateOutputDelta(DOUBLE a_dOutput, UINT32 a_unTeacherValue);
	DOUBLE CalcluateHiddenDelta(DOUBLE a_dOutput, DOUBLE* a_aArray);

public:
	MultiPayerPerceptronNetwork();
	~MultiPayerPerceptronNetwork();

private:
	UINT32**	m_ppPatterns;
	UINT32*		m_pInputPattern;
	UINT32*		m_pOutputPattern;
	UINT32		m_unOutputNodes;
	DOUBLE*		m_aHiddenLayerNodes;
	DOUBLE*		m_aHiddenOutputWeight;
	DOUBLE*		m_aInputHiddenWeight;
	INT			m_nTeachingNodeCount;
	INT			m_nPatternAryCount;
};
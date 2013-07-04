/*==========================================================================
 * Copyright (c) 2001 Carnegie Mellon University.  All Rights Reserved.
 *
 * Use of the Lemur Toolkit for Language Modeling and Information Retrieval
 * is subject to the terms of the software license set forth in the LICENSE
 * file included with this software, and also available at
 * http://www.lemurproject.org/license.html
 *
 *==========================================================================
*/

/*! \page  Text Categorization Evaluation Application within light Lemur toolkit


Usage: TCEval parameter_file

Please refor to the namespace LocalParameter for setting the parameters within the parameter_file

 */


#include "common_headers.hpp"
#include "IndexManager.hpp"
#include "BasicDocStream.hpp"
#include "Param.hpp"
#include "String.hpp"
#include "IndexedReal.hpp"
#include "ScoreAccumulator.hpp"
#include "ResultFile.hpp"
#include "TextQueryRep.hpp"

#define FEATURE_COUNT 7000
#define FEATURE_SELECT false

using namespace lemur::api;

namespace LocalParameter{
  std::string databaseIndex; // the index of the documents
  std::string trainDocs;   // the file of query stream
  std::string testDocs;    // the name of the result file
  std::string resultFile;  // the weighting scheme
  void get() {
    // the string with quotes are the actual variable names to use for specifying the parameters
    databaseIndex    = ParamGetString("index"); 
    trainDocs      = ParamGetString("trainDocs");
    testDocs      = ParamGetString("testDocs");
    resultFile       = ParamGetString("result","res");
  }    
};


void GetAppParam() 
{
  LocalParameter::get();
}

void training(double *legiModel, double *spamModel, double &pSpam, 
	      Index &ind, ifstream &trainIFS)
{}

// This function returns the top k elements accoring to their Mutual Influence score.
// k is defined by the macro FEATURE_COUNT
int *selectMI(ifstream &trainIDFile, Index &ind, int vocabSize)
{
  int *N00, *N01, *N10, *N11;
  map<double,int> termMI;
  
  N00 = new int[vocabSize + 1];
  N01 = new int[vocabSize + 1];
  N10 = new int[vocabSize + 1];
  N11 = new int[vocabSize + 1];
  
  // Initialize the arrays to 0
  for(int i = 0; i <= vocabSize; i++)
  {
	N00[i] = 0;
	N01[i] = 0;
	N10[i] = 0;
	N11[i] = 0;
  }
  
  while (!trainIDFile.eof()){
    int Rel;
    char docIDStr[1000];
    trainIDFile>>docIDStr>>Rel;

	bool *vocabLookup = new bool[vocabSize + 1];
    for(int i = 0; i <= vocabSize; i++)
      vocabLookup[i] = false;
	
    int docID=ind.document(docIDStr);

    TermInfoList *docTermList=ind.termInfoList(docID);
    docTermList->startIteration();
    int count = 0;
    while (docTermList->hasMore()){
      TermInfo *info=docTermList->nextEntry();
      int termFreq=info->count();
      int termID=info->termID();
      
      // Mark if the word has occured in the document
      vocabLookup[termID] = true;
      
      // Add up N11 and N10
      if(Rel == 1)
		N11[termID] += 1;
	  else
	    N10[termID] += 1;
	}
    // For the words that have not occured in the document
	for(int i = 0; i <= vocabSize; i++)
	{
		// Add up all N00 and N01
		if(!vocabLookup[i])
		{
			if (Rel == 1)
				N01[i] += 1;
			else
				N00[i] += 1;
		}
	}
	delete vocabLookup;
	delete docTermList;
  }
  
  // Calculation of MI Score
  for(int i = 0; i <= vocabSize; i++)
  {
	  int N   = N00[i] + N01[i] + N10[i] + N11[i];
	  int N0x = N00[i] + N01[i];
	  int Nx0 = N00[i] + N10[i];
	  int N1x = N10[i] + N11[i];
	  int Nx1 = N01[i] + N11[i];
	  
	  double termMI1 = (double) (N11[i] / N) * log((double) (N * N11[i]) / (N1x * Nx1));
	  double termMI2 = (double) (N01[i] / N) * log((double) (N * N01[i]) / (N0x * Nx1));
	  double termMI3 = (double) (N10[i] / N) * log((double) (N * N10[i]) / (N1x * Nx0));
	  double termMI4 = (double) (N00[i] / N) * log((double) (N * N00[i]) / (N0x * Nx0));
	  
	  double miScore = termMI1 + termMI2 + termMI3 + termMI4;
	  termMI[miScore] = i;cout<<miScore<<" "<<endl;
  }
  
  int *selectedTerms = new int[FEATURE_COUNT];
  int i = 0;
  map<double,int>::reverse_iterator it;
  
  // Put the top k terms in the array selectedTerms
  // k is defined in the macro FEATURE_COUNT
  for(it = termMI.rbegin(); i < FEATURE_COUNT; it++)
  {
	selectedTerms[i++] = it->second;
  }
  
  delete N00;
  delete N01;
  delete N10;
  delete N11;
  
  return selectedTerms;
}

void estTrainModelWithFeatureSelection(ifstream &trainIDFile, double *pWRelModel, double *pWIrrelModel, double &pRel, Index &ind){
  //estimate the naive bayes model from the training data
  int vocabSize=ind.termCountUnique();
  //initiate the value of two models
  for (int t=0; t<=vocabSize; t++){
    pWRelModel[t]=0;
    pWIrrelModel[t]=0;
  }
  
  // Fetch the reduced vocabulary by feature selection through Mutual Influence technique
  int *selectedTerms = selectMI(trainIDFile, ind, vocabSize);
  bool *selectedTermLookup = new bool[vocabSize + 1];

  // Initialize the vocab lookup to false
  for(int i = 0; i <= vocabSize; i++)
	selectedTermLookup[i] = false;

  // Mark all the terms that are part of the new vocabulary as true
  for(int i = 0; i < FEATURE_COUNT; i++)
  {
	  int termIndex = selectedTerms[i];
	  selectedTermLookup[termIndex] = true;
  }
  
  int numTrainDocs=0;         //number of training documents
  int numRelTrainDocs=0;      //number of relevant (i.e.,spam) training documents
  int numWordRelTrainDocs=0;  //number of words in relevant training documents
  int numWordIrrelTrainDocs=0;//number of words in irrelevant training documents

  trainIDFile.clear();
  trainIDFile.seekg(0, ios::beg);
  while (!trainIDFile.eof()){
    int Rel;
    char docIDStr[1000];
    trainIDFile>>docIDStr>>Rel;

    int docID=ind.document(docIDStr);
    numTrainDocs++;

    if (Rel==1){
      /*!!!!!! Implement the code to accumulate the number of relevant training documents !!!!!!*/
      numRelTrainDocs++;
    }

    //go through every document to generate the count of words in each type of document
    TermInfoList *docTermList=ind.termInfoList(docID);
    docTermList->startIteration();
    while (docTermList->hasMore()){
      TermInfo *info=docTermList->nextEntry();
      int termFreq=info->count();
      int termID=info->termID();
      
      if (Rel==1){
	//this is a relevant document
        /*!!!!!!!!!! Implement the code to accumulate term counts for relevant model !!!!!!*/

        // Accumulate the total number of words in relevant documents
        numWordRelTrainDocs += termFreq;
        // Accumulate the term count for the term with id termID in relevant documents if the term is a part of the reduced vocabulary
        if(selectedTermLookup[termID])
			pWRelModel[termID] += termFreq;
			
      }else{
	//this is not a relevant document
        /*!!!!!!!!!! Implement the code to accumulate term counts for irrelevant model !!!!!!*/

        // Accumulate the total number of words in irrelevant documents
        numWordIrrelTrainDocs += termFreq;
        // Accumulate the term count for the term with id termID in irrelevant documents if the term is a part of the reduced vocabulary
        if(selectedTermLookup[termID])
			pWIrrelModel[termID] += termFreq;
      }
    }   
    delete docTermList;
  }

  for (int t=0; t<=vocabSize; t++){
    /*!!!!!! Implement the code to normlize the relevant and irrelevant models (i.e. Sum_wP(w)=1 )  !!!!!!*/
    /*!!!!!! Please use smoothing method !!!!!!*/
    // For relevant document if the term is a part of the reduced vocabulary
    if(selectedTermLookup[t])
		pWRelModel[t] = (1.0 + pWRelModel[t]) / (FEATURE_COUNT + numWordRelTrainDocs);

    // For irrelevant document if the term is a part of the reduced vocabulary
    if(selectedTermLookup[t])
		pWIrrelModel[t] = (1.0 + pWIrrelModel[t]) / (FEATURE_COUNT + numWordIrrelTrainDocs);
  }
  
  /*obtain prior for relevant model (i.e., spam)*/
  pRel=(double)numRelTrainDocs/numTrainDocs;

}

void estTrainModel(ifstream &trainIDFile, double *pWRelModel, double *pWIrrelModel, double &pRel, Index &ind){
  //estimate the naive bayes model from the training data
  int vocabSize=ind.termCountUnique();
  //initiate the value of two models
  for (int t=0; t<=vocabSize; t++){
    pWRelModel[t]=0;
    pWIrrelModel[t]=0;
  }
  
  int numTrainDocs=0;         //number of training documents
  int numRelTrainDocs=0;      //number of relevant (i.e.,spam) training documents
  int numWordRelTrainDocs=0;  //number of words in relevant training documents
  int numWordIrrelTrainDocs=0;//number of words in irrelevant training documents

  while (!trainIDFile.eof()){
    int Rel;
    char docIDStr[1000];
    trainIDFile>>docIDStr>>Rel;

    int docID=ind.document(docIDStr);
    numTrainDocs++;

    if (Rel==1){
      /*!!!!!! Implement the code to accumulate the number of relevant training documents !!!!!!*/
      numRelTrainDocs++;
    }
    

    //go through every document to generate the count of words in each type of document
    TermInfoList *docTermList=ind.termInfoList(docID);
    docTermList->startIteration();
    while (docTermList->hasMore()){
      TermInfo *info=docTermList->nextEntry();
      int termFreq=info->count();
      int termID=info->termID();
      
      if (Rel==1){
	//this is a relevant document
        /*!!!!!!!!!! Implement the code to accumulate term counts for relevant model !!!!!!*/

        // Accumulate the total number of words in relevant documents
        numWordRelTrainDocs += termFreq;
        // Accumulate the term count for the term with id termID in relevant documents
        pWRelModel[termID] += termFreq;

      }else{
	//this is not a relevant document
        /*!!!!!!!!!! Implement the code to accumulate term counts for irrelevant model !!!!!!*/

        // Accumulate the total number of words in irrelevant documents
        numWordIrrelTrainDocs += termFreq;
        // Accumulate the term count for the term with id termID in irrelevant documents
        pWIrrelModel[termID] += termFreq;

      }      
    }   
    delete docTermList;
  }



  for (int t=0; t<=vocabSize; t++){
    /*!!!!!! Implement the code to normlize the relevant and irrelevant models (i.e. Sum_wP(w)=1 )  !!!!!!*/
    /*!!!!!! Please use smoothing method !!!!!!*/
    // For relevant document
    pWRelModel[t] = (1.0 + pWRelModel[t]) / (vocabSize + 1 + numWordRelTrainDocs);

    // For irrelevant document
    pWIrrelModel[t] = (1.0 + pWIrrelModel[t]) / (vocabSize + 1 + numWordIrrelTrainDocs);
  }

  
  /*obtain prior for relevant model (i.e., spam)*/
  pRel=(double)numRelTrainDocs/numTrainDocs;

}

void  printTrainModel(double* pWRelModel, double* pWIrrelModel, double pRel, Index &ind){
  //print out the naive bayes model
  int vocabSize=ind.termCountUnique();
  IndexedRealVector wordVec;
  IndexedRealVector::iterator it;  
  

  cout<<"For Model Prior"<<endl;
  cout<<"Relevant Model:"<<pRel<<"    "<<"Irrelvant Model:"<<1-pRel<<endl;


  double pSumRel=0;
  double pSumIrrel=0;

  for (int t=0; t<=vocabSize; t++){
    pSumRel+=pWRelModel[t];
    pSumIrrel+=pWIrrelModel[t];
  }
  cout<<"Prob Sum is: "<<pSumRel<<" and "<< pSumIrrel<<endl;

  wordVec.clear();
  for (int t=0; t<=vocabSize; t++){
    wordVec.PushValue(t,pWRelModel[t]);
  }
  wordVec.Sort();


  cout<<"Top Words for the Relevant Mode"<<endl;
  int nTopWord=0;
  for (it=wordVec.begin();it!=wordVec.end();it++){
    nTopWord++;
    if (nTopWord>30){
      break;
    }
    cout<<"Top "<<nTopWord<<" "<<ind.term((*it).ind)<<" "<<(*it).val<<endl;
  }

  wordVec.clear();
  for (int t=0; t<=vocabSize; t++){
    wordVec.PushValue(t,pWIrrelModel[t]);
  }
  wordVec.Sort();

  cout<<"Top Words for the Irrelevant Mode"<<endl;
  nTopWord=0;
  for (it=wordVec.begin();it!=wordVec.end();it++){
    nTopWord++;
    if (nTopWord>30){
      break;
    }
    cout<<"Top "<<nTopWord<<" "<<ind.term((*it).ind)<<" "<<(*it).val<<endl;
  }

}



void  getTestRst(ifstream &testIDFile, double* pWRelModel, double* pWIrrelModel, double pRel, IndexedRealVector &results, Index &ind){
  //generate the test results
  int vocabSize=ind.termCountUnique();



  int numTestDoc=0;
  while (!testIDFile.eof()){
    char docIDStr[1000];
    testIDFile>>docIDStr;

    int docID=ind.document(docIDStr);

    double logRelProb=0;  //log probability (i.e., log-likelihood) given relevant model (i.e., spam)
    double logIrrelProb=0;//log probability (i.e., log-likelihood) given irrelvant model (i.e., non-spam)

    TermInfoList *docTermList=ind.termInfoList(docID);
    docTermList->startIteration();
    while (docTermList->hasMore()){
      TermInfo *info=docTermList->nextEntry();
      int termFreq=info->count();
      int termID=info->termID();

      /*!!!!!! Implement the code to accumuate log probability (i.e., log-likelihood) give relevant model and irrelevant model !!!!!!*/
      logRelProb += termFreq * log(pWRelModel[termID]);
      logIrrelProb += termFreq * log(pWIrrelModel[termID]);
    }

    /*Calculate the probability of a document being relevant (i.e. outProb)*/
    /*!!!!!! Please use Bayes Rule; please incoporate the prior probability (i.e., pRel) into the calculating of factor in the next line!!!!!!*/
    double outProb;
    outProb = 1 / (1 + exp(log(1.0 - pRel) + logIrrelProb - (log(pRel) + logRelProb)));

    results.PushValue(docID,outProb);
    numTestDoc++;
  }
}

void printTestRst(ofstream &rstFile, IndexedRealVector &results, Index &ind){
  //print out the test results

  IndexedRealVector::iterator it;  
  for (it=results.begin();it!=results.end();it++){
    rstFile<<ind.document((*it).ind)<<" "<<(*it).val<<endl;
  }

}

/// A retrieval evaluation program
int AppMain(int argc, char *argv[]) {
  

  //Step 1: Open the index file
  Index  *ind;

  try {
    ind  = IndexManager::openIndex(LocalParameter::databaseIndex);
  } 
  catch (Exception &ex) {
    ex.writeMessage();
    throw Exception("RelEval", "Can't open index, check parameter index");
  }

  //Step 2: Open the id file to get training and test documents
  ifstream trainIDFile;
  try {
    trainIDFile.open(LocalParameter::trainDocs.c_str());
  } 
  catch (Exception &ex) {
    ex.writeMessage(cerr);
    throw Exception("NBClassify", 
                    "Can't open train Document Files");
  }

  ifstream testIDFile;
  try {
    testIDFile.open(LocalParameter::testDocs.c_str());
  } 
  catch (Exception &ex) {
    ex.writeMessage(cerr);
    throw Exception("NBClassify", 
                    "Can't open test Document Files");
  }

  ofstream rstFile;
  try {
    rstFile.open(LocalParameter::resultFile.c_str());
  } 
  catch (Exception &ex) {
    ex.writeMessage(cerr);
    throw Exception("NBClassify", 
                    "Can't open result Files");
  }

  //Step 3: Training process to generate model parameters
  int vocabSize=ind->termCountUnique();
  double pWRelModel[vocabSize+1]; //p(W|Relevant Docs); for documents with "1" (spam)
  double pWIrrelModel[vocabSize+1]; //p(W|Irrelevant Docs); for documents with "0" (non-spam)
  double pRel;       //probability of relevant model (i.e., spam model)       


  if(FEATURE_SELECT)
    estTrainModelWithFeatureSelection(trainIDFile, pWRelModel, pWIrrelModel, pRel, *ind);
  else
	estTrainModel(trainIDFile, pWRelModel, pWIrrelModel, pRel, *ind);
  printTrainModel(pWRelModel, pWIrrelModel, pRel, *ind);


  //Step 4: Test the performance
  IndexedRealVector results;

  results.clear();
  getTestRst(testIDFile, pWRelModel, pWIrrelModel, pRel, results, *ind); 

  printTestRst(rstFile, results, *ind);

  delete ind;
  return 0;
}

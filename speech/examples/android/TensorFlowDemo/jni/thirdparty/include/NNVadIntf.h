/************************************************************************/
/** \file NNVadIntf.h
 * \n
 * \n
 * Written by @Author, @Date\n
 * \n
 * 
 * Copyright ZTSpeech 2014
 * www.ztspeech.com
 * 
 */
/************************************************************************/

#ifndef __NNVAD_INTF_H__
#define __NNVAD_INTF_H__

#ifdef __cplusplus
extern "C" {
#endif


#ifndef STC_SOURCE_COMPILE
#ifndef STC_DYNAMIC_LOAD
//#pragma comment(lib, "NNVad.lib")
#endif
#endif

#define VAD_SAMPLE_RATEKHZ	16
#define VAD_SAMPLE_RATE (VAD_SAMPLE_RATEKHZ*1000)
#define	VAD_FRAME_LENMS	10
#define VAD_FRAME_SIZE	(VAD_SAMPLE_RATEKHZ*VAD_FRAME_LENMS)	// shorts, 10ms of 16k16bit mono pcm

#define VAD_MAX_LENGTH_SEC	60
#define VAD_MAX_FRAME_NUM	(VAD_MAX_LENGTH_SEC*1000/VAD_FRAME_LENMS)

/* VAD work mode */
#define VAD_MODE_DNN		0
#define VAD_MODE_ENERGY		1
#define VAD_MODE_DNNENERGY	2
#define VAD_MODE_WEBRTC		3


//#define	NNVAPI	__stdcall
#define	NNVAPI

/************************************************************************/
/** 
 * frame-based vad                                        
 */

typedef void	*NNV_HANDLE;

#define VD_PARAM_ESTIFRAMENUM	11
#define VD_PARAM_MINENERGY		12
#define	VD_PARAM_MAXENERGY		13
#define	VD_PARAM_WORKMODE		14
#define	VD_PARAM_WEBRTCMODE		15
#define	VD_PARAM_CTXFRAMENUM	16
#define	VD_PARAM_BASERANGE      17
#define	VD_PARAM_MINDYNARANGE   18
#define	VD_PARAM_MAXDYNARANGE   19
#define VD_PARAM_MINAECENERGY	20
#define VD_PARAM_MINDYNAENERGY	21

int NNVAPI VAD_SysInit();
int NNVAPI VAD_SysExit();

NNV_HANDLE NNVAPI NNV_NewVad(int sampleRate, int frameLenMs, int mode);
int NNVAPI NNV_ResetVad(NNV_HANDLE hVad);
int NNVAPI NNV_RestartVad(NNV_HANDLE hVad);
int NNVAPI NNV_DelVad(NNV_HANDLE hVad);
int NNVAPI NNV_SetVadParam(NNV_HANDLE hVad, int nParam, void *pVal);

int NNVAPI NNV_InputWave(NNV_HANDLE hVad, char *pWaveData,
		int nFrameNum, int bIsEnd, int nIsAec);
int NNVAPI NNV_InputFloatWave(NNV_HANDLE hVad, float *pWaveData,
		int nFrameNum, int bIsEnd, int nIsAec);
int NNVAPI NNV_OffsetFrame(NNV_HANDLE hVad, int nFrame);
int NNVAPI NNV_GetMaxFrame(NNV_HANDLE hVad);
float NNVAPI NNV_GetFrameProb(NNV_HANDLE hVad, int nFrame, int nDim);
float NNVAPI NNV_GetLastFrameEnergy(NNV_HANDLE hVad);
float NNVAPI NNV_GetThresholdEnergy(NNV_HANDLE hVad);

/*
 internal use only, don't call.
 */
int NNVAPI NNV_Freeze(NNV_HANDLE hVad, int bFreeze, int nIsAec);
int NNVAPI NNV_FreezeSpeech(NNV_HANDLE hVad, int bFreeze, int nFrameNum);


/************************************************************************/
/** 
 * sentence-based vad                                        
 */

typedef void	*VD_HANDLE;

    /* frame number added before vad start */
#define VD_PARAM_PREFRAMENUM	1
    /* vad start frame number */
#define VD_PARAM_MINVOCFRAMENUM	2
    /* vad start frame ratio */
#define VD_PARAM_MINVOCRATIO	5
    /* minimal frame number of one sentence */
#define VD_PARAM_MINSPEECHFRAMENUM	6
    /* maximal frame number of one sentence */
#define VD_PARAM_MAXSPEECHFRAMENUM	7
    /* vad stop frame number */
#define VD_PARAM_MINSILFRAMENUM	3
    /* vad stop frame ratio */
#define VD_PARAM_MINSILRATIO	4
    /* enable pitch detection */
#define VD_PARAM_ENBLEPITCH     8
#define VD_PARAM_BEGINPITCH_FRAMENUM   9
#define VD_PARAM_ENDPITCH_FRAMENUM     10

VD_HANDLE NNVAPI VD_NewVad(int nMode);
int NNVAPI VD_ResetVad(VD_HANDLE hVad);
int NNVAPI VD_RestartVad(VD_HANDLE hVad);
int NNVAPI VD_DelVad(VD_HANDLE hVad);
int NNVAPI VD_SetVadParam(VD_HANDLE hVad, int nParam, void *pVal);

int NNVAPI VD_InputWave(VD_HANDLE hVad, const short *pWaveData, int nSampleNum, int bIsEnd, int nIsAec);
int NNVAPI VD_InputFloatWave(VD_HANDLE hVad, const float *pWaveData, int nSampleNum, int bIsEnd, int nIsAec);
int NNVAPI VD_GetOffsetFrame(VD_HANDLE hVad);
int NNVAPI VD_SetStart(VD_HANDLE hVad, int nIsAec);
int NNVAPI VD_GetVoiceStartFrame(VD_HANDLE hVad);
int NNVAPI VD_GetVoiceStopFrame(VD_HANDLE hVad);
int NNVAPI VD_GetVoiceFrameNum(VD_HANDLE hVad);
const short * NNVAPI VD_GetVoice(VD_HANDLE hVad);
const float * NNVAPI VD_GetFloatVoice(VD_HANDLE hVad);
int NNVAPI VD_Freeze(NNV_HANDLE hVad, int bFreeze, int nSampleNum);
float NNVAPI VD_GetLastFrameEnergy(NNV_HANDLE hVad);
float NNVAPI VD_GetThresholdEnergy(NNV_HANDLE hVad);


#ifdef __cplusplus
};
#endif

#endif

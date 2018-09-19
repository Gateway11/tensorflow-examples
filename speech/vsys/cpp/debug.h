//
//  debug.h
//  vsys
//
//  Created by 薯条 on 2018/1/20.
//  Copyright © 2018年 薯条. All rights reserved.
//

#ifndef DEBUG_H
#define DEBUG_H

#ifndef LOG_TAG
#define LOG_TAG NULL
#endif

#if 1
#if defined(__ANDROID__) || defined(ANDROID)
#include <android/log.h>
#define VSYS_DEBUGD(...) __android_log_print(ANDROID_LOG_DEBUG, LOG_TAG, __VA_ARGS__)
#define VSYS_DEBUGI(...) __android_log_print(ANDROID_LOG_INFO, LOG_TAG, __VA_ARGS__)
#define VSYS_DEBUGE(...) __android_log_print(ANDROID_LOG_ERROR, LOG_TAG, __VA_ARGS__)
#else
#include <stdio.h>
#include <unistd.h>
#include <sys/time.h>
#define VSYS_DEBUGD(...){ __dx_logtime_print('D', LOG_TAG); printf(__VA_ARGS__); printf("\n");}
#define VSYS_DEBUGI(...){ __dx_logtime_print('I', LOG_TAG); printf(__VA_ARGS__); printf("\n");}
#define VSYS_DEBUGE(...){ __dx_logtime_print('E', LOG_TAG); printf(__VA_ARGS__); printf("\n");}

#define __dx_logtime_print(level, tag) \
    struct timeval tv; \
    struct tm ltm; \
    gettimeofday(&tv, NULL); \
    localtime_r(&tv.tv_sec, &ltm); \
    printf("%02d-%02d %02d:%02d:%02d.%03d  %04d %c %s: ", \
    ltm.tm_mon, ltm.tm_mday, \
    ltm.tm_hour, ltm.tm_min, ltm.tm_sec, \
    tv.tv_usec / 1000, getpid(), level, tag);
#endif
#else
#define VSYS_DEBUGD(...)
#define VSYS_DEBUGI(...)
#define VSYS_DEBUGE(...)
#endif

#endif /* DEBUG_H */

//
//  buf_manager.h
//  vsys
//
//  Created by 薯条 on 2018/1/18.
//  Copyright © 2018年 薯条. All rights reserved.
//

#ifndef BUF_MANAGER_H
#define BUF_MANAGER_H

#include <string.h>
#include <cstddef>

static inline void release_buffer(void* p);

static inline void* malloc_buffer(const uint32_t length){
    return length > 0 ? new char[length] : NULL;
}

static inline float** malloc_buffer2(const uint32_t row, const uint32_t length){
    float* temp = (float *)malloc_buffer(row * length * sizeof(float));
    if(!temp) return NULL;
    float** p = new float*[row];
    if(!p){
        release_buffer(temp);
        return NULL;
    }
    for (uint32_t i = 0; i < row; i++)
        p[i] = temp + i * length;
    return p;
}

static inline void release_buffer(void* p){
    if(p) delete[] (char *)p;
}

static inline void release_buffer2(void** p){
    if(p){
        release_buffer(p[0]);
        delete[] (float **)p;
    }
}

static inline bool remalloc_buffer(float** p, const uint32_t ofset, const uint32_t size){
    float* temp = (float *)malloc_buffer(size * sizeof(float));
    if(!temp) {
        release_buffer(*p);
        *p = NULL;
        return false;
    }
    if(*p){
        memcpy(temp, *p, ofset * sizeof(float));
        release_buffer(*p);
    }
    *p = temp;
    return true;
}

static inline bool remalloc_buffer2(float*** p,
                                    const uint32_t row, const uint32_t ofset, const uint32_t length){
    float** temp  = malloc_buffer2(row, length);
    if(!temp){
        release_buffer2((void **)(*p));
        *p = NULL;
        return false;
    }
    if(*p){
        for (uint32_t i = 0; i < row; i++)
            memcpy(temp[i], (*p)[i], ofset * sizeof(float));
        release_buffer2((void **)(*p));
    }
    *p = temp;
    return true;
}

static inline bool write_to_buffer(float** buff, uint32_t* ofset,
                                   uint32_t* total_size, const float *data, const uint32_t length){
    if(!length) return false;
    if((length + *ofset) > *total_size){
        *total_size = length + *ofset;
        remalloc_buffer(buff, *ofset, *total_size);
    }
    memcpy(*buff + *ofset, data, length * sizeof(float));
    *ofset += length;
    return true;
}

static inline bool write_to_buffer(float*** buff, const uint32_t row, uint32_t* ofset,
                                    uint32_t* total_size, const float** data, const uint32_t length){
    if(!length) return false;
    if((length + *ofset) > *total_size){
        *total_size = length + *ofset;
        remalloc_buffer2(buff, row, *ofset, *total_size);
    }
    for (uint32_t i = 0; i < row; i++)
        memcpy((*buff)[i] + *ofset, data[i], length * sizeof(float));
    *ofset += length;
    return true;
}

#endif /* BUF_MANAGER_H */

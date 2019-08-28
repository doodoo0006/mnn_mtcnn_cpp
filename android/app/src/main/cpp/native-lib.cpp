#include <jni.h>
#include "opencv2/opencv.hpp"
#include "imgProcess.h"

extern "C"
JNIEXPORT jobject JNICALL
Java_opencv4unity_camera1gltest1_MyNDKOpencv_structFromNative(JNIEnv *env, jclass type)
{

    jclass cSructInfo=env->FindClass("opencv4unity/camera1gltest1/ResultFromJni");
    jfieldID text=env->GetFieldID(cSructInfo,"text","Ljava/lang/String;");
    jfieldID number=env->GetFieldID(cSructInfo,"num","I");

    jobject oStructInfo=env->AllocObject(cSructInfo);
    env->SetIntField(oStructInfo,number,888);
    std::string fuc = "fuck the Jni: ";
    jstring jstrn=env->NewStringUTF(fuc.c_str());
    env->SetObjectField(oStructInfo,text,jstrn);

    return oStructInfo;
}

extern "C" 
JNIEXPORT jobject JNICALL 
Java_opencv4unity_camera1gltest1_MyNDKOpencv_getScannerEffect(JNIEnv *env, jclass type,
                                                              jintArray pixels_, 
                                                              jstring fileDir_, 
                                                              jint w, jint h, jint model) 
{
    jclass cSructInfo=env->FindClass("opencv4unity/camera1gltest1/ResultFromJni2");
    jfieldID cXLoc=env->GetFieldID(cSructInfo,"x","I");
    jfieldID cYLoc=env->GetFieldID(cSructInfo,"y","I");
    jfieldID cResultInt=env->GetFieldID(cSructInfo,"resultInt","[I");
    jobject oStructInfo=env->AllocObject(cSructInfo);

    const char* testfilePath = NULL;
    testfilePath = env->GetStringUTFChars(fileDir_, 0);
    if(testfilePath == NULL )
    {
        return NULL;
    }

    unsigned char tmpjb = 0;
    jint *pixels = env->GetIntArrayElements(pixels_,  &tmpjb);
    if(pixels==NULL)
    {
        return NULL;
    }

    cv::Mat imgData(h, w, CV_8UC4, pixels);
    deal(imgData, testfilePath);

    int size = w * h;
    jintArray result = env->NewIntArray(size);
    env->SetIntArrayRegion(result, 0, size, pixels);
    env->SetObjectField(oStructInfo,cResultInt,result);

    env->ReleaseIntArrayElements(pixels_, pixels, 0);
    return oStructInfo;
}


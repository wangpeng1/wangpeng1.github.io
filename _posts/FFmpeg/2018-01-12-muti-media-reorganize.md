---
layout: post
title: 多媒体整理
category: FFMPEG
tags: ffmpeg
description: 主要包含FFMPEG编解码，speex编解码，librtmp，libyuv，直播录播
---
## FFMPEG

http://blog.csdn.net/jiangwei0910410003/article/details/17466369  生成JIN Signature

http://www.ihubin.com/blog/android-ffmpeg-demo-1/

http://blog.csdn.net/quan648997767/article/details/70172166 android studio

http://blog.csdn.net/mabeijianxi/article/details/72983362  android studio c++

https://www.cnblogs.com/CoderTian/p/6651343.html android studio c++

http://blog.csdn.net/mabeijianxi/article/details/72983362 合成音视频更多接口

https://developer.android.google.cn/ndk/guides/android_mk.html#over

http://blog.csdn.net/u014418171/article/details/53337759 修改运行的bug

https://www.jianshu.com/p/ceaa286d8aff 线程 x64

http://blog.csdn.net/u014418171/article/details/53337759 线程回调，其他选项 硬件加速

https://proandroiddev.com/android-ndk-how-to-integrate-pre-built-libraries-in-case-of-the-ffmpeg-7ff24551a0f

线程启动 多次启动，硬编码
https://voiddog.github.io/2017/06/24/%E7%BB%99FFmpeg%E5%8A%A0%E4%B8%8AMediaCodec%E5%92%8C%E7%BA%BF%E7%A8%8B%E6%94%AF%E6%8C%81/

https://www.jianshu.com/p/d26e7d788c0e 硬编码和录制视频合成

https://github.com/LaiFeng-Android/SopCastComponent 硬解码录制合成


https://www.cnblogs.com/tinywan/p/6337504.html ffmpeg直接摄像头

https://www.cnblogs.com/lknlfy/archive/2012/03/31/2426788.html ffmpeg 捕获一针

https://github.com/saki4510t/AudioVideoRecordingSample 混合录合

http://www.jb51.net/article/129642.htm android硬编码 路径合并

https://github.com/lakeinchina/librestreaming 推流 硬编码 GPU 图像处理

https://github.com/LaiFeng-Android/SopCastComponent

https://www.cnblogs.com/raomengyang/p/6288019.html libyuv

http://www.cnblogs.com/raomengyang/p/6544908.html 录屏 以及相关参数说明 rtmp推流

http://blog.csdn.net/zjqlovell/article/details/52446750 视频高清参数


##Speex

https://github.com/i-p-tel/sipdroid speex 编解码

https://www.jianshu.com/p/e74700dd07cf speex降噪 回升消除

https://www.cnblogs.com/jiangu66/p/3172118.html 回声消除

http://blog.csdn.net/CAZICAQUW/article/details/7333186 回声消除

http://blog.csdn.net/qq_29078329/article/details/56287338 回声消除

https://github.com/i-p-tel/sipdroid  rtm speex

http://www.360doc.com/content/13/0613/22/2036337_292678527.shtml speex参数说明

https://github.com/dakatso/SpeexExample 回声消除的库

http://blog.csdn.net/ywl5320/article/details/78503768  Android通过OpenSL ES播放音频套路详解

https://github.com/mars-ma/Android-OpenSLES-Demo/tree/master/app/src/main  OpenSL ES speex

https://github.com/CL-window/audio_speex speex简单的参数流程说明

https://github.com/dengzhi00/SpeexAndroid/blob/master/jni/SpeexAndroid.cpp speex参数说明

https://github.com/fghjhuang/WifiRealTimeSpeeker speex编解码的使用 和参数的使用说明 一些byte字节转化

https://github.com/walid1992/AndroidSpeexDenoise 降噪处理，简单的调用使用 speex编解码

https://github.com/springtom/AudioPCMSpeexFLV-android speex 回声消除，播放控制

https://github.com/BlueTel/RtpDemo rtp发送

 
##指针

https://wenku.baidu.com/view/2c55a61d227916888486d794.html

https://wenku.baidu.com/view/953fb977a26925c52cc5bf8e.html

https://portal.tacc.utexas.edu/c/document_library/get_file?uuid=268c81d1-2cf5-41ec-8641-e110d59e77c7&groupId=13601

http://www.studytonight.com/c/command-line-argument.php

## FreeSwith

http://blog.csdn.net/thrill008/article/details/78203686

##SIP PS 流
https://blog.csdn.net/yingyemin/article/details/82910366
https://blog.csdn.net/Zhu__/article/details/78907004
https://blog.csdn.net/NBA_1/article/details/80172125

https://www.telestax.com/blog/jain-sip-stack-for-android/

## 编译android-libjpeg-turbo
1.https://github.com/openstf/android-libjpeg-turbo 参考此篇文章，然后修改编译文件，将静态库改成share库 生成so
2.然后将此库https://github.com/JavaNoober/Light 复制或者替换掉原来的就可以编译最新的arm64-v8a

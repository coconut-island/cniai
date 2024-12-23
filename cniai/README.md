# CNIAI

## Dev Dependencies && Env

## os

Ubuntu 22.04.5 LTS (Jammy Jellyfish)

### chip

NVIDIA GeForce RTX 4070 Ti

### ffmpeg

```text
https://github.com/FFmpeg/FFmpeg/archive/refs/tags/n5.1.4.zip
../configure --prefix=/usr/local/ffmpeg --enable-shared --enable-pic --disable-debug --disable-x86asm --disable-doc
```

/usr/local/ffmpeg

### cuda

cuda-12.1
/usr/local/cuda

### TensorRT

TensorRT-10.2.0.19
/usr/local/cuda

### Video_Codec_SDK

Video_Codec_SDK_11.1.5
/usr/local/nvcodec/include
/usr/local/nvcodec/lib

### opencv

4.5.4

### rtsp server

```text
docker run --rm -it -e MTX_PROTOCOLS=tcp -e MTX_WEBRTCADDITIONALHOSTS=192.168.x.x -p 8554:8554 -p 1935:1935 -p 8888:8888 -p 8889:8889 -p 8890:8890/udp -p 8189:8189/udp bluenviron/mediamtx:1.10.0
```
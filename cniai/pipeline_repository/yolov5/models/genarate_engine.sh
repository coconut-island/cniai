
trtexec --onnx=models/yolov5s_dynamic.onnx --minShapes=images:1x3x320x320 --optShapes=images:16x3x640x640 --maxShapes=images:32x3x1280x1280 --saveEngine=models/yolov5s_dynamic.trt --fp16
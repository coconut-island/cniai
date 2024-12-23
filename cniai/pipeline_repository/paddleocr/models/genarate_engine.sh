
trtexec --onnx=./models/ch_PP-OCRv4_det_infer.onnx --minShapes=x:1x3x32x32 --optShapes=x:16x3x640x640 --maxShapes=x:32x3x1280x1280 --saveEngine=models/ch_PP-OCRv4_det_infer.trt --fp16

trtexec --onnx=./models/ch_PP-OCRv4_rec_infer.onnx --minShapes=x:1x3x48x32 --optShapes=x:16x3x48x960 --maxShapes=x:32x3x48x2048 --saveEngine=models/ch_PP-OCRv4_rec_infer.trt --fp16

trtexec --onnx=./models/ch_ppocr_mobile_v2.0_cls_infer.onnx --minShapes=x:1x3x48x32 --optShapes=x:16x3x48x640 --maxShapes=x:32x3x48x1024 --saveEngine=models/ch_ppocr_mobile_v2.0_cls_infer.trt --fp16

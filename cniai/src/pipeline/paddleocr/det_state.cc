#include "det_state.h"

#include <cniai_cuda_kernel/imgproc.h>
#include <opencv2/opencv.hpp>

#include "common/logging.h"

namespace cniai::pipeline::paddleocr {

void PaddleocrDetExecutorState::preprocess(
    ExecutorStatePreprocessParam *executorStatePreprocessParam) {
    auto requests = executorStatePreprocessParam->requests;
    auto &runtime = executorStatePreprocessParam->runtime;
    auto profileIndex = executorStatePreprocessParam->profileIndex;
    auto inputTensorMap = executorStatePreprocessParam->inputTensorMap;
    auto &cudaStream = executorStatePreprocessParam->cudaStream;

    auto inputName = runtime->getEngine().getIOTensorName(0);
    auto minHeight =
        (int)runtime->getEngine()
            .getProfileShape(inputName, profileIndex, nvinfer1::OptProfileSelector::kMIN)
            .d[2];
    auto minWidth =
        (int)runtime->getEngine()
            .getProfileShape(inputName, profileIndex, nvinfer1::OptProfileSelector::kMIN)
            .d[3];
    auto maxHeight =
        (int)runtime->getEngine()
            .getProfileShape(inputName, profileIndex, nvinfer1::OptProfileSelector::kMAX)
            .d[2];
    auto maxWidth =
        (int)runtime->getEngine()
            .getProfileShape(inputName, profileIndex, nvinfer1::OptProfileSelector::kMAX)
            .d[3];

    int imageMaxWidth = minWidth;
    int imageMaxHeight = minHeight;
    for (auto &request : *requests) {
        auto &frame = reinterpret_cast<PaddleocrDetRequest *>(request)->frame;
        imageMaxWidth = std::max(imageMaxWidth, frame.getWidth());
        imageMaxHeight = std::max(imageMaxHeight, frame.getHeight());
    }

    int inputWidth = std::clamp(imageMaxWidth, minWidth, maxWidth);
    int inputHeight = std::clamp(imageMaxHeight, minHeight, maxHeight);

    inputWidth = std::max(int(round(float(inputWidth) / 32) * 32), 32);
    inputHeight = std::max(int(round(float(inputHeight) / 32) * 32), 32);

    CNIAI_CHECK(!(*inputTensorMap).empty());
    auto &inputTensor = (*inputTensorMap).begin()->second;
    inputTensor->reshape(
        ITensor::makeShape({requests->size(), 3, static_cast<std::size_t>(inputHeight),
                            static_cast<std::size_t>(inputWidth)}));

    for (int b = 0; b < requests->size(); ++b) {
        auto *request = reinterpret_cast<PaddleocrDetRequest *>((*requests)[b]);
        auto &frame = request->frame;

        auto imageWidth = frame.getWidth();
        auto imageHeight = frame.getHeight();

        int imgW;
        int imgH;
        int padW = 0;
        int padH = 0;
        if (imageWidth <= inputWidth && imageHeight <= inputHeight) {
            imgW = imageWidth;
            imgH = imageHeight;
        } else {
            float gain = std::min(float(inputWidth) / float(imageWidth),
                                  float(inputHeight) / float(imageHeight));
            imgW = static_cast<int>(gain * float(imageWidth));
            imgH = static_cast<int>(gain * float(imageHeight));
        }

        cniai_cuda_kernel::imgproc::rgbResizeBilinearPadNorm(
            (uint8_t *)frame.data(), (float *)inputTensor->getBatchPointer(b), imageWidth,
            imageHeight, imgW, imgH, inputWidth, inputHeight, padW, padH,
            mJsonConfig["pad0"], mJsonConfig["pad1"], mJsonConfig["pad2"],
            mJsonConfig["scale"], mJsonConfig["mean0"], mJsonConfig["mean1"],
            mJsonConfig["mean2"], mJsonConfig["std0"], mJsonConfig["std1"],
            mJsonConfig["std2"], true, true, cudaStream->get());
    }
}

void PaddleocrDetExecutorState::postprocess(
    ExecutorStatePostprocessParam *executorStatePostprocessParam) {
    auto requests = executorStatePostprocessParam->requests;
    auto responses = executorStatePostprocessParam->responses;
    auto &runtime = executorStatePostprocessParam->runtime;
    auto profileIndex = executorStatePostprocessParam->profileIndex;
    auto outputTensorMap = executorStatePostprocessParam->outputTensorMap;
    auto outputHostTensorMap = executorStatePostprocessParam->outputHostTensorMap;
    auto &cudaStream = executorStatePostprocessParam->cudaStream;

    LOG_DEBUG("responses size: {}", responses->size());
    cudaStream->synchronize();

    auto inputName = runtime->getEngine().getIOTensorName(0);
    auto minHeight =
        (int)runtime->getEngine()
            .getProfileShape(inputName, profileIndex, nvinfer1::OptProfileSelector::kMIN)
            .d[2];
    auto minWidth =
        (int)runtime->getEngine()
            .getProfileShape(inputName, profileIndex, nvinfer1::OptProfileSelector::kMIN)
            .d[3];
    auto maxHeight =
        (int)runtime->getEngine()
            .getProfileShape(inputName, profileIndex, nvinfer1::OptProfileSelector::kMAX)
            .d[2];
    auto maxWidth =
        (int)runtime->getEngine()
            .getProfileShape(inputName, profileIndex, nvinfer1::OptProfileSelector::kMAX)
            .d[3];

    int imageMaxWidth = minWidth;
    int imageMaxHeight = minHeight;
    for (auto &request : *requests) {
        auto &frame = reinterpret_cast<PaddleocrDetRequest *>(request)->frame;
        imageMaxWidth = std::max(imageMaxWidth, frame.getWidth());
        imageMaxHeight = std::max(imageMaxHeight, frame.getHeight());
    }

    int inputWidth = std::clamp(imageMaxWidth, minWidth, maxWidth);
    int inputHeight = std::clamp(imageMaxHeight, minHeight, maxHeight);

    inputWidth = std::max(int(round(float(inputWidth) / 32) * 32), 32);
    inputHeight = std::max(int(round(float(inputHeight) / 32) * 32), 32);

    auto &outputHostTensor = (*outputHostTensorMap).begin()->second;
    auto outputShape = outputHostTensor->getShape();

    double detDbThresh = mJsonConfig["detDbThresh"];
    double detDbBoxThresh = mJsonConfig["detDbBoxThresh"];
    double detDbUnclipRatio = mJsonConfig["detDbUnclipRatio"];
    std::string detDbScoreMode = mJsonConfig["detDbScoreMode"];

    for (int b = 0; b < outputShape.d[0]; ++b) {
        auto *request = reinterpret_cast<PaddleocrDetRequest *>((*requests)[b]);
        auto &frame = request->frame;

        auto outputHostPtr = (float *)outputHostTensor->getBatchPointer(b);

        int n2 = outputShape.d[2];
        int n3 = outputShape.d[3];
        int nc = n2 * n3;

        std::vector<float> pred(nc, 0.0);
        std::vector<unsigned char> cbuf(nc, ' ');

        for (int i = 0; i < nc; i++) {
            pred[i] = float(outputHostPtr[i]);
            cbuf[i] = (unsigned char)((outputHostPtr[i]) * 255);
        }

        cv::Mat cbufMap(n2, n3, CV_8UC1, (unsigned char *)cbuf.data());
        cv::Mat predMap(n2, n3, CV_32F, (float *)pred.data());

        const double threshold = detDbThresh * 255;
        const double maxvalue = 255;
        cv::Mat bitMap;
        cv::threshold(cbufMap, bitMap, threshold, maxvalue, cv::THRESH_BINARY);
        cv::Mat dilationMap;
        cv::Mat dilaEle = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(2, 2));
        cv::dilate(bitMap, dilationMap, dilaEle);

        std::vector<std::vector<std::vector<int>>> boxes = mPostProcessor.BoxesFromBitmap(
            predMap, bitMap, detDbBoxThresh, detDbUnclipRatio, detDbScoreMode);

        auto imageWidth = frame.getWidth();
        auto imageHeight = frame.getHeight();

        float gain = (imageWidth <= inputWidth && imageHeight <= inputHeight)
                         ? 1.0f
                         : std::min(float(inputWidth) / float(imageWidth),
                                    float(inputHeight) / float(imageHeight));

        for (int n = 0; n < boxes.size(); n++) {
            boxes[n] = mPostProcessor.OrderPointsClockwise(boxes[n]);
            for (int m = 0; m < boxes[0].size(); m++) {
                boxes[n][m][0] /= gain;
                boxes[n][m][1] /= gain;

                boxes[n][m][0] =
                    int(std::min(std::max(boxes[n][m][0], 0), imageWidth - 1));
                boxes[n][m][1] =
                    int(std::min(std::max(boxes[n][m][1], 0), imageHeight - 1));
            }
        }

        std::vector<std::vector<std::vector<int>>> rootPoints;
        for (auto &box : boxes) {
            int rectWidth, rectHeight;
            rectWidth =
                int(sqrt(pow(box[0][0] - box[1][0], 2) + pow(box[0][1] - box[1][1], 2)));
            rectHeight =
                int(sqrt(pow(box[0][0] - box[3][0], 2) + pow(box[0][1] - box[3][1], 2)));
            if (rectWidth <= 4 || rectHeight <= 4)
                continue;
            rootPoints.push_back(box);
        }

        boxes = rootPoints;

        auto response = new PaddleocrDetResponse();
        response->request = request;
        for (auto &box : boxes) {
            PaddleOCR::OCRPredictResult res;
            res.box = box;
            response->oCRPredictResults.push_back(res);
        }

        PaddleOCR::Utility::sorted_boxes(response->oCRPredictResults);

        (*responses).emplace_back(reinterpret_cast<ExecutorResponse *>(response));
    }

    LOG_DEBUG("responses size: {}", responses->size());

    LOG_DEBUG("POSE PROCESS DONE");
}

void PaddleocrDetExecutorState::initialize() { initConfig(); }

void PaddleocrDetExecutorState::initConfig() {
    mJsonConfig["inputName"] = "x";
    mJsonConfig["outputName"] = "sigmoid_0.tmp_0";

    mJsonConfig["scale"] = 1.0f / 255.0f;
    mJsonConfig["mean0"] = 0.485f;
    mJsonConfig["mean1"] = 0.456f;
    mJsonConfig["mean2"] = 0.406f;
    mJsonConfig["std0"] = 1 / 0.229f;
    mJsonConfig["std1"] = 1 / 0.224f;
    mJsonConfig["std2"] = 1 / 0.225f;
    mJsonConfig["pad0"] =
        (128.0f * (float)mJsonConfig["scale"] - (float)mJsonConfig["mean0"]) *
        (float)mJsonConfig["std0"];
    mJsonConfig["pad1"] =
        (128.0f * (float)mJsonConfig["scale"] - (float)mJsonConfig["mean1"]) *
        (float)mJsonConfig["std1"];
    mJsonConfig["pad2"] =
        (128.0f * (float)mJsonConfig["scale"] - (float)mJsonConfig["mean2"]) *
        (float)mJsonConfig["std2"];

    mJsonConfig["detDbThresh"] = 0.3;
    mJsonConfig["detDbBoxThresh"] = 0.5;
    mJsonConfig["detDbUnclipRatio"] = 2.0;
    mJsonConfig["detDbScoreMode"] = "slow";
    mJsonConfig["useDilation"] = false;
}

void showDetResponse(CniFrame &frame, PaddleocrDetResponse *response) {
    auto hostFrame = frame.clone(CniFrame::MemoryType::CPU);

    cv::Mat img(hostFrame.getHeight(), hostFrame.getWidth(), CV_8UC3, hostFrame.data());

    cv::Mat imgVis;
    cv::cvtColor(img, img, cv::COLOR_RGB2BGR);
    img.copyTo(imgVis);
    for (auto &ocrResult : response->oCRPredictResults) {
        cv::Point rookPoints[4];
        for (int m = 0; m < ocrResult.box.size(); m++) {
            rookPoints[m] = cv::Point(int(ocrResult.box[m][0]), int(ocrResult.box[m][1]));
        }

        const cv::Point *ppt[1] = {rookPoints};
        int npt[] = {4};
        cv::polylines(imgVis, ppt, npt, 1, 1, CV_RGB(0, 255, 0), 2, 8, 0);
    }

    for (auto &ocrResult : response->oCRPredictResults) {
        auto box = ocrResult.box;

        int dstWidth =
            int(sqrt(pow(box[0][0] - box[1][0], 2) + pow(box[0][1] - box[1][1], 2)));
        int dstHeight =
            int(sqrt(pow(box[0][0] - box[3][0], 2) + pow(box[0][1] - box[3][1], 2)));

        cv::Point2f dstPoints[4];
        dstPoints[0] = cv::Point2f(0., 0.);
        dstPoints[1] = cv::Point2f(dstWidth, 0.);
        dstPoints[2] = cv::Point2f(dstWidth, dstHeight);
        dstPoints[3] = cv::Point2f(0.f, dstHeight);

        cv::Point2f srcPoints[4];
        unsigned char *d_dst;
        cudaMallocManaged(&d_dst, dstWidth * dstHeight * 3);
        if (float(dstHeight) >= float(dstWidth) * 1.5) {
            srcPoints[1] = cv::Point2f(box[0][0], box[0][1]);
            srcPoints[0] = cv::Point2f(box[1][0], box[1][1]);
            srcPoints[3] = cv::Point2f(box[2][0], box[2][1]);
            srcPoints[2] = cv::Point2f(box[3][0], box[3][1]);
            std::swap(dstWidth, dstHeight);
            for (auto &dstPoint : dstPoints) {
                std::swap(dstPoint.x, dstPoint.y);
            }
        } else {
            srcPoints[0] = cv::Point2f(box[0][0], box[0][1]);
            srcPoints[1] = cv::Point2f(box[1][0], box[1][1]);
            srcPoints[2] = cv::Point2f(box[2][0], box[2][1]);
            srcPoints[3] = cv::Point2f(box[3][0], box[3][1]);
        }

        cv::Mat M = cv::getPerspectiveTransform(srcPoints, dstPoints);
        float matrix[9];
        for (int i = 0; i < 9; ++i) {
            matrix[i] = static_cast<float>(M.at<double>(i / 3, i % 3));
        }

        cniai_cuda_kernel::imgproc::warpPerspective(
            (uint8_t *)frame.data(), d_dst, frame.getWidth(), frame.getHeight(), dstWidth,
            dstHeight, matrix[0], matrix[1], matrix[2], matrix[3], matrix[4], matrix[5],
            matrix[6], matrix[7], matrix[8], nullptr);

        cudaDeviceSynchronize();

        cv::Mat dst_img(dstHeight, dstWidth, CV_8UC3, d_dst);
        cv::cvtColor(dst_img, dst_img, cv::COLOR_RGB2BGR);

        cv::namedWindow("Display window"); // Create a window for display.
        cv::imshow("Display window", dst_img);
        cv::waitKey(0);

        {
            int imgWidth = dstWidth;
            int imgHeight = dstHeight;

            int inputWidth = 330;
            int inputHeight = 48;
            unsigned char *d_dst;
            cudaMallocManaged(&d_dst, inputWidth * inputHeight * 3);

            float gain = std::min(float(inputWidth) / float(imgWidth),
                                  float(inputHeight) / float(imgHeight));
            int imgW = static_cast<int>(gain * float(imgWidth));
            int imgH = static_cast<int>(gain * float(imgHeight));
            int padW = 0;
            int padH = 0;
            //            padH = (inputHeight - imgH) / 2;
            //            imgH = 48;

            cniai_cuda_kernel::imgproc::warpPerspectiveRgbResizeBilinearPad(
                (uint8_t *)frame.data(), d_dst, frame.getWidth(), frame.getHeight(),
                dstWidth, dstHeight, imgW, imgH, inputWidth, inputHeight, 0, 0, 0,
                matrix[0], matrix[1], matrix[2], matrix[3], matrix[4], matrix[5],
                matrix[6], matrix[7], matrix[8], true, false, true, nullptr);

            cudaDeviceSynchronize();
            unsigned char *dd_dst;
            cudaMallocManaged(&dd_dst, inputWidth * inputHeight * 3);

            cniai_cuda_kernel::imgproc::rgbPlanarPackedSwap(d_dst, dd_dst, inputWidth,
                                                            inputHeight, nullptr);

            cudaDeviceSynchronize();

            cv::Mat dst_img(inputHeight, inputWidth, CV_8UC3, dd_dst);
            cv::cvtColor(dst_img, dst_img, cv::COLOR_RGB2BGR);

            cv::namedWindow("Display window"); // Create a window for display.
            cv::imshow("Display window", dst_img);
            cv::waitKey(0);
        }
    }

    cv::namedWindow("Display window"); // Create a window for display.
    cv::imshow("Display window", imgVis);
    cv::waitKey(0);
}

} // namespace cniai::pipeline::paddleocr
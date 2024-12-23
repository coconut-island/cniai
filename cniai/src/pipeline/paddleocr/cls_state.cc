#include "cls_state.h"

#include <cniai_cuda_kernel/imgproc.h>
#include <opencv2/opencv.hpp>

#include "common/logging.h"
#include "utility.h"

namespace cniai::pipeline::paddleocr {

void PaddleocrClsExecutorState::preprocess(
    ExecutorStatePreprocessParam *executorStatePreprocessParam) {
    auto requests = executorStatePreprocessParam->requests;
    auto &runtime = executorStatePreprocessParam->runtime;
    auto profileIndex = executorStatePreprocessParam->profileIndex;
    auto inputTensorMap = executorStatePreprocessParam->inputTensorMap;
    auto &cudaStream = executorStatePreprocessParam->cudaStream;

    auto inputName = runtime->getEngine().getIOTensorName(0);
    auto minWidth =
        (int)runtime->getEngine()
            .getProfileShape(inputName, profileIndex, nvinfer1::OptProfileSelector::kMIN)
            .d[3];
    auto maxWidth =
        (int)runtime->getEngine()
            .getProfileShape(inputName, profileIndex, nvinfer1::OptProfileSelector::kMAX)
            .d[3];

    int inputHeight = inputTensorMap->begin()->second->getShape().d[2];

    int inputMaxWidth = minWidth;
    for (auto &_request : *requests) {
        auto *request = reinterpret_cast<PaddleocrClsRequest *>(_request);
        auto &ocrPredictResult = request->ocrPredictResult;
        auto &box = ocrPredictResult.box;
        int warpPerspectiveImageWidth =
            int(sqrt(pow(box[0][0] - box[1][0], 2) + pow(box[0][1] - box[1][1], 2)));
        int warpPerspectiveHeight =
            int(sqrt(pow(box[0][0] - box[3][0], 2) + pow(box[0][1] - box[3][1], 2)));
        if (float(warpPerspectiveHeight) >= float(warpPerspectiveImageWidth) * 1.5) {
            std::swap(warpPerspectiveImageWidth, warpPerspectiveHeight);
        }
        inputMaxWidth = std::max(inputMaxWidth, int((float)warpPerspectiveImageWidth /
                                                    warpPerspectiveHeight * inputHeight));
    }
    int inputWidth = std::clamp(inputMaxWidth, minWidth, maxWidth);

    auto &inputTensor = (*inputTensorMap).begin()->second;
    inputTensor->reshape(
        ITensor::makeShape({requests->size(), 3, static_cast<std::size_t>(inputHeight),
                            static_cast<std::size_t>(inputWidth)}));

    for (int b = 0; b < requests->size(); ++b) {
        auto *request = reinterpret_cast<PaddleocrClsRequest *>((*requests)[b]);
        auto &frame = request->frame;
        auto ocrPredictResult = request->ocrPredictResult;
        auto &box = ocrPredictResult.box;

        int warpPerspectiveImageWidth =
            int(sqrt(pow(box[0][0] - box[1][0], 2) + pow(box[0][1] - box[1][1], 2)));
        int warpPerspectiveImageHeight =
            int(sqrt(pow(box[0][0] - box[3][0], 2) + pow(box[0][1] - box[3][1], 2)));

        cv::Point2f dstPoints[4];
        dstPoints[0] = cv::Point2f(0., 0.);
        dstPoints[1] = cv::Point2f(warpPerspectiveImageWidth, 0.);
        dstPoints[2] = cv::Point2f(warpPerspectiveImageWidth, warpPerspectiveImageHeight);
        dstPoints[3] = cv::Point2f(0.f, warpPerspectiveImageHeight);

        cv::Point2f srcPoints[4];
        if (float(warpPerspectiveImageHeight) >= float(warpPerspectiveImageWidth) * 1.5) {
            srcPoints[1] = cv::Point2f(box[0][0], box[0][1]);
            srcPoints[0] = cv::Point2f(box[1][0], box[1][1]);
            srcPoints[3] = cv::Point2f(box[2][0], box[2][1]);
            srcPoints[2] = cv::Point2f(box[3][0], box[3][1]);
            std::swap(warpPerspectiveImageWidth, warpPerspectiveImageHeight);
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

        int imgH = inputHeight;
        float ratio =
            float(warpPerspectiveImageWidth) / float(warpPerspectiveImageHeight);
        int imgW = inputWidth;
        if (ceilf(imgH * ratio) < inputWidth) {
            imgW = int(ceilf(inputHeight * ratio));
        }

        cniai_cuda_kernel::imgproc::warpPerspectiveRgbResizeBilinearPadNorm(
            (uint8_t *)frame.data(), (float *)inputTensor->getBatchPointer(b),
            frame.getWidth(), frame.getHeight(), warpPerspectiveImageWidth,
            warpPerspectiveImageHeight, imgW, imgH, inputWidth, inputHeight,
            mJsonConfig["pad0"], mJsonConfig["pad1"], mJsonConfig["pad2"],
            mJsonConfig["scale"], mJsonConfig["mean0"], mJsonConfig["mean1"],
            mJsonConfig["mean2"], mJsonConfig["std0"], mJsonConfig["std1"],
            mJsonConfig["std2"], matrix[0], matrix[1], matrix[2], matrix[3], matrix[4],
            matrix[5], matrix[6], matrix[7], matrix[8], true, true, false,
            cudaStream->get());
    }
}

void PaddleocrClsExecutorState::postprocess(
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

    auto &outputHostTensor = (*outputHostTensorMap).begin()->second;
    auto outputShape = outputHostTensor->getShape();

    for (int b = 0; b < outputShape.d[0]; ++b) {
        auto *request = reinterpret_cast<PaddleocrClsRequest *>((*requests)[b]);
        auto ocrPredictResult = request->ocrPredictResult;
        auto outputHostPtr = (float *)outputHostTensor->getBatchPointer(b);
        int label = int(PaddleOCR::Utility::argmax(&outputHostPtr[0], &outputHostPtr[2]));
        auto score = float(*std::max_element(&outputHostPtr[0], &outputHostPtr[2]));
        auto response = new PaddleocrClsResponse();
        response->oCRPredictResult = ocrPredictResult;
        response->oCRPredictResult.cls_label = label;
        response->oCRPredictResult.cls_score = score;
        (*responses).emplace_back(reinterpret_cast<ExecutorResponse *>(response));
    }
}

void PaddleocrClsExecutorState::initialize() { initConfig(); }

void PaddleocrClsExecutorState::initConfig() {
    mJsonConfig["scale"] = 1.0f / 255.0f;

    mJsonConfig["mean0"] = 0.5f;
    mJsonConfig["mean1"] = 0.5f;
    mJsonConfig["mean2"] = 0.5f;

    mJsonConfig["std0"] = 1 / 0.5f;
    mJsonConfig["std1"] = 1 / 0.5f;
    mJsonConfig["std2"] = 1 / 0.5f;

    mJsonConfig["pad0"] = 0.f;
    mJsonConfig["pad1"] = 0.f;
    mJsonConfig["pad2"] = 0.f;
}

} // namespace cniai::pipeline::paddleocr
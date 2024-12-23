#include "yolov5_detect_state.h"

#include <cniai_cuda_kernel/imgproc.h>

namespace cniai::pipeline::yolov5_detect {

float calculateIoU(const DetectionBox &box1, const DetectionBox &box2) {
    int x1 = std::max(box1.mX, box2.mX);
    int y1 = std::max(box1.mY, box2.mY);
    int x2 = std::min(box1.mX + box1.mW, box2.mX + box2.mW);
    int y2 = std::min(box1.mY + box1.mH, box2.mY + box2.mH);

    float intersection = std::max(0.0f, static_cast<float>(x2 - x1)) *
                         std::max(0.0f, static_cast<float>(y2 - y1));
    float unionArea = (static_cast<float>(box1.mW) * static_cast<float>(box1.mH)) +
                      (static_cast<float>(box2.mW) * static_cast<float>(box2.mH)) -
                      intersection;
    return intersection / unionArea;
}

void nonMaximumSuppression(std::vector<DetectionBox> &boxes, float iouThreshold) {

    std::sort(
        boxes.begin(), boxes.end(),
        [](const DetectionBox &a, const DetectionBox &b) { return a.mScore > b.mScore; });

    for (size_t i = 0; i < boxes.size(); ++i) {
        if (boxes[i].mScore == 0)
            continue;

        for (size_t j = i + 1; j < boxes.size(); ++j) {

            if (boxes[i].mClassId != boxes[j].mClassId)
                continue;

            if (calculateIoU(boxes[i], boxes[j]) > iouThreshold)
                boxes[j].mScore = 0;
        }
    }

    boxes.erase(std::remove_if(boxes.begin(), boxes.end(),
                               [](const DetectionBox &box) { return box.mScore == 0; }),
                boxes.end());
}

void clipBox(DetectionBox &box, int imageWidth, int imageHeight) {
    box.mX = std::max(0, box.mX);
    box.mY = std::max(0, box.mY);

    box.mW = std::min(box.mW, imageWidth - box.mX);
    box.mH = std::min(box.mH, imageHeight - box.mY);

    if (box.mX + box.mW > imageWidth) {
        box.mW = imageWidth - box.mX;
    }
    if (box.mY + box.mH > imageHeight) {
        box.mH = imageHeight - box.mY;
    }
}

void Yolov5DetectExecutorState::preprocess(
    ExecutorStatePreprocessParam *executorStatePreprocessParam) {
    auto requests = executorStatePreprocessParam->requests;
    auto inputTensorMap = executorStatePreprocessParam->inputTensorMap;
    auto &cudaStream = executorStatePreprocessParam->cudaStream;

    int inputWidth = mJsonConfig["inputWidth"];
    int inputHeight = mJsonConfig["inputHeight"];

    CNIAI_CHECK(!(*inputTensorMap).empty());
    auto &inputTensor = (*inputTensorMap).begin()->second;
    inputTensor->reshape(
        ITensor::makeShape({requests->size(), 3, static_cast<std::size_t>(inputHeight),
                            static_cast<std::size_t>(inputWidth)}));

    for (int b = 0; b < requests->size(); ++b) {
        auto *request = reinterpret_cast<Yolov5DetectRequest *>((*requests)[b]);
        auto &frame = request->frame;

        auto imageWidth = frame.getWidth();
        auto imageHeight = frame.getHeight();

        float gain = std::min(float(inputWidth) / float(imageWidth),
                              float(inputHeight) / float(imageHeight));
        int imgW = static_cast<int>(gain * float(imageWidth));
        int imgH = static_cast<int>(gain * float(imageHeight));
        int padW = (inputWidth - imgW) / 2;
        int padH = (inputHeight - imgH) / 2;

        CNIAI_CHECK(frame.getMemoryType() == CniFrame::MemoryType::GPU);
        CNIAI_CHECK(frame.getFormat() == CniFrame::Format::RGB ||
                    frame.getFormat() == CniFrame::Format::NV12);

        if (frame.getFormat() == CniFrame::Format::RGB) {
            cniai_cuda_kernel::imgproc::rgbResizeBilinearPadNorm(
                (uint8_t *)frame.data(), (float *)inputTensor->getBatchPointer(b),
                imageWidth, imageHeight, imgW, imgH, inputWidth, inputHeight, padW, padH,
                mJsonConfig["pad0"], mJsonConfig["pad1"], mJsonConfig["pad2"],
                mJsonConfig["scale"], mJsonConfig["mean0"], mJsonConfig["mean1"],
                mJsonConfig["mean2"], mJsonConfig["std0"], mJsonConfig["std1"],
                mJsonConfig["std2"], true, false, cudaStream->get());
        }

        if (frame.getFormat() == CniFrame::Format::NV12) {
            cniai_cuda_kernel::imgproc::nv12ToRgbResizeBilinearPadNorm(
                (uint8_t *)frame.data(), (float *)inputTensor->getBatchPointer(b),
                imageWidth, imageHeight, imgW, imgH, inputWidth, inputHeight, padW, padH,
                mJsonConfig["pad0"], mJsonConfig["pad1"], mJsonConfig["pad2"],
                mJsonConfig["scale"], mJsonConfig["mean0"], mJsonConfig["mean1"],
                mJsonConfig["mean2"], mJsonConfig["std0"], mJsonConfig["std1"],
                mJsonConfig["std2"], true, false, cudaStream->get());
        }
    }
}

void Yolov5DetectExecutorState::postprocess(
    ExecutorStatePostprocessParam *executorStatePostprocessParam) {
    auto requests = executorStatePostprocessParam->requests;
    auto responses = executorStatePostprocessParam->responses;
    auto outputTensorMap = executorStatePostprocessParam->outputTensorMap;
    auto outputHostTensorMap = executorStatePostprocessParam->outputHostTensorMap;
    auto &cudaStream = executorStatePostprocessParam->cudaStream;

    LOG_DEBUG("responses size: {}", responses->size());
    cudaStream->synchronize();

    auto outputShape = (*outputHostTensorMap)[mJsonConfig["outputName"]]->getShape();

    for (int b = 0; b < outputShape.d[0]; ++b) {
        auto *request = reinterpret_cast<Yolov5DetectRequest *>((*requests)[b]);
        auto &frame = request->frame;

        auto outputHostPtr =
            (float *)(*outputHostTensorMap)[mJsonConfig["outputName"]]->getBatchPointer(
                b);
        auto anchors = outputShape.d[1];
        auto cls = outputShape.d[2];
        int m_nm = 0;
        int nc = cls - m_nm - 5;
        int mi = 5 + nc;

        std::vector<DetectionBox> boxes;

        for (int i = 0; i < anchors; ++i) {
            auto anchorPtr = outputHostPtr + i * cls;
            float score = anchorPtr[4];
            if (score <= mJsonConfig["confThreshold"]) {
                continue;
            }
            float center_x = anchorPtr[0];
            float center_y = anchorPtr[1];
            float width = anchorPtr[2];
            float height = anchorPtr[3];
            float x1 = center_x - width / 2;
            float y1 = center_y - height / 2;
            float x2 = center_x + width / 2;
            float y2 = center_y + height / 2;

            float maxClassScore = 0;
            float classScore;
            int maxClassId = 0;
            for (int j = 5; j < mi; ++j) {
                classScore = anchorPtr[j] * score;
                if (classScore > maxClassScore) {
                    maxClassScore = classScore;
                    maxClassId = j - 5;
                }
            }

            std::string className =
                std::string(mJsonConfig["classNames"][std::to_string(maxClassId)]);

            auto box =
                DetectionBox(x1, y1, x2 - x1, y2 - y1, score, maxClassId, className);
            boxes.emplace_back(box);
        }

        nonMaximumSuppression(boxes, mJsonConfig["iouThreshold"]);

        int inputWidth = mJsonConfig["inputWidth"];
        int inputHeight = mJsonConfig["inputHeight"];

        auto imageWidth = frame.getWidth();
        auto imageHeight = frame.getHeight();

        float gain = std::min(float(inputWidth) / float(imageWidth),
                              float(inputHeight) / float(imageHeight));
        int imgW = static_cast<int>(gain * float(imageWidth));
        int imgH = static_cast<int>(gain * float(imageHeight));
        int padW = (inputWidth - imgW) / 2;
        int padH = (inputHeight - imgH) / 2;

        for (auto &box : boxes) {
            if (box.mScore == 0) {
                CNIAI_CHECK(box.mScore == 0);
                continue;
            }

            box.mX = static_cast<int>(static_cast<float>(box.mX - padW) / gain);
            box.mY = static_cast<int>(static_cast<float>(box.mY - padH) / gain);
            box.mW = static_cast<int>(static_cast<float>(box.mW + padW) / gain);
            box.mH = static_cast<int>(static_cast<float>(box.mH + padH) / gain);

            clipBox(box, frame.getWidth(), frame.getHeight());
            LOG_DEBUG("box: {}", box.toString());
        }

        LOG_DEBUG("box size: {}", boxes.size());

        auto response = new Yolov5DetectResponse();
        response->detectBoxes = boxes;
        (*responses).emplace_back(reinterpret_cast<ExecutorResponse *>(response));
    }

    LOG_DEBUG("responses size: {}", responses->size());

    LOG_DEBUG("POSE PROCESS DONE");
}

void Yolov5DetectExecutorState::initialize() { initConfig(); }

void Yolov5DetectExecutorState::initConfig() {
    mJsonConfig["inputWidth"] = 640;
    mJsonConfig["inputHeight"] = 640;
    mJsonConfig["pad0"] = 114.0f / 255.0f;
    mJsonConfig["pad1"] = 114.0f / 255.0f;
    mJsonConfig["pad2"] = 114.0f / 255.0f;
    mJsonConfig["scale"] = 1.0f / 255.0f;
    mJsonConfig["mean0"] = 0.0f;
    mJsonConfig["mean1"] = 0.0f;
    mJsonConfig["mean2"] = 0.0f;
    mJsonConfig["std0"] = 1.0f;
    mJsonConfig["std1"] = 1.0f;
    mJsonConfig["std2"] = 1.0f;

    mJsonConfig["confThreshold"] = 0.25;
    mJsonConfig["iouThreshold"] = 0.45;

    mJsonConfig["inputName"] = "images";
    mJsonConfig["outputName"] = "output0";

    mJsonConfig["classNames"] = json::parse(R"({
            "0": "person",
            "1": "bicycle",
            "2": "car",
            "3": "motorcycle",
            "4": "airplane",
            "5": "bus",
            "6": "train",
            "7": "truck",
            "8": "boat",
            "9": "traffic light",
            "10": "fire hydrant",
            "11": "stop sign",
            "12": "parking meter",
            "13": "bench",
            "14": "bird",
            "15": "cat",
            "16": "dog",
            "17": "horse",
            "18": "sheep",
            "19": "cow",
            "20": "elephant",
            "21": "bear",
            "22": "zebra",
            "23": "giraffe",
            "24": "backpack",
            "25": "umbrella",
            "26": "handbag",
            "27": "tie",
            "28": "suitcase",
            "29": "frisbee",
            "30": "skis",
            "31": "snowboard",
            "32": "sports ball",
            "33": "kite",
            "34": "baseball bat",
            "35": "baseball glove",
            "36": "skateboard",
            "37": "surfboard",
            "38": "tennis racket",
            "39": "bottle",
            "40": "wine glass",
            "41": "cup",
            "42": "fork",
            "43": "knife",
            "44": "spoon",
            "45": "bowl",
            "46": "banana",
            "47": "apple",
            "48": "sandwich",
            "49": "orange",
            "50": "broccoli",
            "51": "carrot",
            "52": "hot dog",
            "53": "pizza",
            "54": "donut",
            "55": "cake",
            "56": "chair",
            "57": "couch",
            "58": "potted plant",
            "59": "bed",
            "60": "dining table",
            "61": "toilet",
            "62": "tv",
            "63": "laptop",
            "64": "mouse",
            "65": "remote",
            "66": "keyboard",
            "67": "cell phone",
            "68": "microwave",
            "69": "oven",
            "70": "toaster",
            "71": "sink",
            "72": "refrigerator",
            "73": "book",
            "74": "clock",
            "75": "vase",
            "76": "scissors",
            "77": "teddy bear",
            "78": "hair drier",
            "79": "toothbrush"
        })");
}

} // namespace cniai::pipeline::yolov5_detect
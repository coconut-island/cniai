#pragma once

#include <string>

#include "common/assert.h"
#include "common/str_util.h"

namespace cniai::pipeline::data_struct {

class DetectionBox {

public:
    int mX;
    int mY;
    int mW;
    int mH;
    float mScore;
    int mClassId;
    std::string mClassName;

public:
    explicit DetectionBox(int x, int y, int w, int h, float score, int classId)
        : mX(x), mY(y), mW(w), mH(h), mScore(score), mClassId(classId) {}

    explicit DetectionBox(int x, int y, int w, int h, float score, int classId,
                          std::string &className)
        : mX(x), mY(y), mW(w), mH(h), mScore(score), mClassId(classId),
          mClassName(className) {}

    [[nodiscard]] std::string toString() const {
        std::stringstream ss;
        ss << "DetectionBox: { "
           << "X: " << mX << ", "
           << "Y: " << mY << ", "
           << "W: " << mW << ", "
           << "H: " << mH << ", "
           << "Score: " << mScore << ", "
           << "ClassId: " << mClassId;
        if (!mClassName.empty()) {
            ss << ", ClassName: " << mClassName;
        }
        ss << " }";
        return ss.str();
    }
};

} // namespace cniai::pipeline::data_struct

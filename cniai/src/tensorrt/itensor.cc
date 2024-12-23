#include "tensorrt/itensor.h"

#include "common/str_util.h"

#include <initializer_list>
#include <memory>

namespace cniai::tensorrt {

nvinfer1::Dims ITensor::makeShape(std::initializer_list<std::size_t> const &dims) {
    CNIAI_CHECK_WITH_INFO(dims.size() <= nvinfer1::Dims::MAX_DIMS,
                          "Number of dimensions is too large");
    nvinfer1::Dims shape{};
    shape.nbDims = static_cast<decltype(Shape::nbDims)>(dims.size());
    std::copy(dims.begin(), dims.end(), shape.d);
    return shape;
}

std::string ITensor::toString(nvinfer1::Dims const &dims) {
    if (dims.nbDims < 0) {
        return "invalid";
    } else if (dims.nbDims == 0) {
        return "()";
    } else {
        return cniai::str_util::arr2str(dims.d, dims.nbDims);
    }
}

std::size_t ITensor::getSingleBatchSizeInBytes() {
    std::size_t singleBatchSizeInBytes = 0;
    auto &shape = getShape();
    for (int i = 0; i < shape.nbDims; ++i) {
        if (i == 0) {
            singleBatchSizeInBytes = 1;
        } else {
            CNIAI_CHECK(shape.d[i] > 0);
            singleBatchSizeInBytes = singleBatchSizeInBytes * shape.d[i];
        }
    }
    singleBatchSizeInBytes = singleBatchSizeInBytes * getDTypeSize(getDataType());
    return singleBatchSizeInBytes;
}

void *ITensor::getBatchPointer(int batchIndex) {
    CNIAI_CHECK_WITH_INFO(getShape().nbDims > 0 && batchIndex < getShape().d[0] &&
                              batchIndex >= 0,
                          "Shape: {}, batchIndex: {}", toString(getShape()), batchIndex);

    return ((uint8_t *)data()) + (batchIndex * getSingleBatchSizeInBytes());
}

} // namespace cniai::tensorrt
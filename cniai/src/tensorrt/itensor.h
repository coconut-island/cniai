#pragma once

#include "tensorrt/ibuffer.h"

#include <NvInferRuntime.h>

#include <algorithm>
#include <cassert>
#include <cstdint>
#include <functional>
#include <initializer_list>
#include <limits>
#include <memory>
#include <numeric>
#include <ostream>
#include <string>
#include <type_traits>

namespace cniai::tensorrt {

class ITensor : virtual public IBuffer {
public:
    using Shape = nvinfer1::Dims;
    using DimType = std::remove_reference_t<decltype(Shape::d[0])>;

    ~ITensor() override = default;

    //!
    //! \brief Returns the tensor dimensions.
    //!
    [[nodiscard]] virtual Shape const &getShape() const = 0;

    //!
    //! \brief Sets the tensor dimensions. The new size of the tensor will be
    //! `volume(dims)`
    //!
    virtual void reshape(Shape const &dims) = 0;

    void resize(std::size_t newSize) override {
        if (newSize == getSize())
            return;

        reshape(makeShape({static_cast<unsigned long>(castSize(newSize))}));
    }

    //!
    //! \brief Not allowed to copy.
    //!
    ITensor(ITensor const &) = delete;

    //!
    //! \brief Not allowed to copy.
    //!
    ITensor &operator=(ITensor const &) = delete;

    //!
    //! \brief Returns the volume of the dimensions. Returns -1 if `d.nbDims < 0`.
    //!
    static std::int64_t volume(Shape const &dims) {
        {
            return dims.nbDims < 0 ? -1
                   : dims.nbDims == 0
                       ? 0
                       : std::accumulate(dims.d, dims.d + dims.nbDims, std::int64_t{1},
                                         std::multiplies<>{});
        }
    }

    //!
    //! \brief Returns the volume of the dimensions. Throws if `d.nbDims < 0`.
    //!
    static std::size_t volumeNonNegative(Shape const &shape) {
        auto const vol = volume(shape);
        CNIAI_CHECK_WITH_INFO(0 <= vol, "Invalid tensor shape");
        return static_cast<std::size_t>(vol);
    }

    //!
    //! \brief A convenience function to create a tensor shape with the given dimensions.
    //!
    static Shape makeShape(std::initializer_list<std::size_t> const &dims);

    //!
    //! \brief A convenience function for converting a tensor shape to a `string`.
    //!
    static std::string toString(Shape const &dims);

    //!
    //! \brief A convenience function to compare shapes.
    //!
    static bool shapeEquals(Shape const &lhs, Shape const &rhs) {
        return shapeEquals(lhs, rhs.d, rhs.nbDims);
    }

    //!
    //! \brief A convenience function to compare shapes.
    //!
    template <typename T>
    static bool shapeEquals(Shape const &lhs, T const *dims, std::size_t count) {
        return lhs.nbDims == count && std::equal(lhs.d, lhs.d + lhs.nbDims, dims);
    }

    [[nodiscard]] bool shapeEquals(Shape const &other) const {
        return shapeEquals(getShape(), other);
    }

    [[nodiscard]] bool
    shapeEquals(std::initializer_list<std::size_t> const &other) const {
        return shapeEquals(getShape(), other.begin(), other.size());
    }

    template <typename T>
    bool shapeEquals(T const *dims, std::size_t count) const {
        return shapeEquals(getShape(), dims, count);
    }

    std::size_t getSingleBatchSizeInBytes();

    void *getBatchPointer(int batchIndex);

protected:
    ITensor() = default;

    static DimType castSize(std::size_t newSize) {
        CNIAI_CHECK_WITH_INFO(newSize <= std::numeric_limits<DimType>::max(),
                              "New size is too large. Use reshape() instead.");
        return static_cast<DimType>(newSize);
    }
};

} // namespace cniai::tensorrt

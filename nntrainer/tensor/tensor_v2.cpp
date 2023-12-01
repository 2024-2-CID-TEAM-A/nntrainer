// SPDX-License-Identifier: Apache-2.0
/**
 * @file	tensor_v2.cpp
 * @date	01 December 2023
 * @brief	This is a TensorV2 class
 * @see		https://github.com/nnstreamer/nntrainer
 * @author	Jijoong Moon <jijoong.moon@samsung.com>
 * @author	Donghyeon Jeong <dhyeon.jeong@samsung.com>
 * @bug		No known bugs except for NYI items
 */

#include <float_tensor.h>
#include <half_tensor.h>
#include <tensor_v2.h>

namespace nntrainer {

TensorV2::TensorV2(std::string name_, Tformat fm, Tdatatype d_type) {
  if (d_type == Tdatatype::FP32) {
    itensor = new FloatTensor(name_, fm);
  } else if (d_type == Tdatatype::FP16) {
#ifdef ENABLE_FP16
    itensor = new HalfTensor(name_, fm);
#else
    throw std::invalid_argument("Error: enable-fp16 is not enabled");
#endif
  }
}

void TensorV2::allocate() { itensor->allocate(); }

void TensorV2::deallocate() { itensor->deallocate(); }

bool TensorV2::isAllocated() { return itensor->isAllocated(); }

size_t TensorV2::getIndex(unsigned int b, unsigned int c, unsigned int h,
                          unsigned int w) const noexcept {
  return itensor->getIndex(b, c, h, w);
}

void TensorV2::setValue(float value) { itensor->setValue(value); }

void TensorV2::setValue(unsigned int b, unsigned int c, unsigned int h,
                        unsigned int w, float value) {
  itensor->setValue(b, c, h, w, value);
}

void TensorV2::setZero() { itensor->setZero(); }

void TensorV2::initialize() { itensor->initialize(); }

void TensorV2::initialize(TensorBase::Initializer init) {
  itensor->initialize(init);
}

void TensorV2::print(std::ostream &out) const { itensor->print(out); }

} // namespace nntrainer

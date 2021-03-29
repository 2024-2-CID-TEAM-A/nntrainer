// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2020 Jijoong Moon <jijoong.moon@samsung.com>
 *
 * @file	rnn.cpp
 * @date	17 March 2021
 * @brief	This is Recurrent Layer Class of Neural Network
 * @see		https://github.com/nnstreamer/nntrainer
 * @author	Jijoong Moon <jijoong.moon@samsung.com>
 * @bug		No known bugs except for NYI items
 *
 */

#include <cmath>
#include <layer_internal.h>
#include <lazy_tensor.h>
#include <nntrainer_error.h>
#include <nntrainer_log.h>
#include <parse_util.h>
#include <rnn.h>
#include <util_func.h>

namespace nntrainer {

const std::string RNNLayer::type = "rnn";

// - weight_xh ( input to hidden )
//  : [1, 1, input_size, unit (hidden_size) ]
// - weight_hh ( hidden to hidden )
//  : [1, 1, unit (hidden_size) , unit (hidden_size)]
// - bias_h ( hidden bias )
//  : [1, 1, 1, unit (hidden_size)]
enum RNNParams { weight_xh, weight_hh, bias_h };

int RNNLayer::initialize(Manager &manager) {
  int status = ML_ERROR_NONE;
  if (getNumInputs() != 1) {
    throw std::invalid_argument("RNN layer takes only one input");
  }

  // input_dim = [ batch, 1, time_iterantion, feature_size ]
  // outut_dim = [ batch, 1, time_iteration, hidden_size ( unit ) ]
  output_dim[0] = input_dim[0];
  output_dim[0].width(unit);

  TensorDim bias_dim = TensorDim();
  bias_dim.setTensorDim(3, unit);

  TensorDim dim_xh = output_dim[0];
  dim_xh.height(input_dim[0].width());
  dim_xh.batch(1);

  TensorDim dim_hh = output_dim[0];
  dim_hh.height(unit);
  dim_hh.batch(1);

  if (weights.empty()) {
    weights.reserve(3);
    // weight_initializer can be set sepeartely. weight_xh initializer,
    // weight_hh initializer kernel initializer & recurrent_initializer in keras
    // for now, it is set same way.
    weights.emplace_back(dim_xh, weight_initializer, weight_regularizer,
                         weight_regularizer_constant, true, "RNN:weight_xh");
    weights.emplace_back(dim_hh, weight_initializer, weight_regularizer,
                         weight_regularizer_constant, true, "RNN:weight_hh");
    weights.emplace_back(bias_dim, bias_initializer, WeightRegularizer::NONE,
                         1.0f, true, "RNN:bias_h");
    manager.trackWeights(weights);
  } else {
    weights[RNNParams::weight_xh].reset(dim_xh, weight_initializer,
                                        weight_regularizer,
                                        weight_regularizer_constant, true);
    weights[RNNParams::weight_hh].reset(dim_hh, weight_initializer,
                                        weight_regularizer,
                                        weight_regularizer_constant, true);
    weights[RNNParams::bias_h].reset(bias_dim, bias_initializer,
                                     WeightRegularizer::NONE, 1.0f, true);
  }

  bias_dim.batch(input_dim[0].batch());
  h_prev = Tensor(bias_dim);
  h_prev.setZero();

  return status;
}

void RNNLayer::setProperty(const PropertyType type, const std::string &value) {
  int status = ML_ERROR_NONE;
  switch (type) {
  case PropertyType::unit: {
    if (!value.empty()) {
      status = setUint(unit, value);
      throw_status(status);
      output_dim[0].width(unit);
    }
    break;
  case PropertyType::activation:
    if (!value.empty()) {
      ActivationType acti_type = (ActivationType)parseType(value, TOKEN_ACTI);
      Layer::activation_type = acti_type;
      acti_func.setActiFunc(acti_type);
    }
    break;
  default:
    Layer::setProperty(type, value);
    break;
  }
  }
}

void RNNLayer::forwarding(bool training) {
  Tensor &weight_xh =
    weightAt(static_cast<int>(RNNParams::weight_xh)).getVariableRef();
  Tensor &weight_hh =
    weightAt(static_cast<int>(RNNParams::weight_hh)).getVariableRef();
  Tensor &bias_h =
    weightAt(static_cast<int>(RNNParams::bias_h)).getVariableRef();

  Tensor &hidden_ = net_hidden[0]->getVariableRef();
  Tensor &input_ = net_input[0]->getVariableRef();

  Tensor temp;
  Tensor hs_prev;
  Tensor hs;

  // TODO : check merge b and t index
  for (unsigned int b = 0; b < input_dim[0].batch(); ++b) {
    Tensor islice = input_.getBatchSlice(b, 1);
    Tensor oslice = hidden_.getBatchSlice(b, 1);

    for (unsigned int t = 0; t < islice.height(); ++t) {
      Tensor xs = input_.getSharedDataTensor(TensorDim(1, 1, 1, islice.width()),
                                             t * islice.width());
      // Calculate Hidden
      // activation(xs.dot(weight_xh).add(hs_prev.dot(weight_hh).add(bias_h)));
      hs = oslice.getSharedDataTensor(TensorDim(1, 1, 1, oslice.width()),
                                      t * oslice.width());
      if (t > 0) {
        hs_prev = oslice.getSharedDataTensor(TensorDim(1, 1, 1, oslice.width()),
                                             (t - 1) * oslice.width());
      } else {
        hs_prev = h_prev.getBatchSlice(b, 1);
      }

      hs_prev.dot(weight_hh, temp);

      xs.dot(weight_xh, hs);
      temp.add_i(bias_h);

      hs.add_i(temp);
      // TODO : In-place calculation for activation
      acti_func.run_fn(hs, hs);
    }
    h_prev.getBatchSlice(b, 1).copy(hs);
  }
}

void RNNLayer::copy(std::shared_ptr<Layer> l) {
  Layer::copy(l);

  std::shared_ptr<RNNLayer> from = std::static_pointer_cast<RNNLayer>(l);
  this->unit = from->unit;
}

void RNNLayer::calcDerivative() {
  //  NYI
}

void RNNLayer::calcGradient() {
  // NYI
}

} // namespace nntrainer
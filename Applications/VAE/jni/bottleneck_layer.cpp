// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2024 Daniel Jang <minhyukjang@snu.ac.kr>
 *
 * @file   upsample_layer.cpp
 * @brief  
 * @see    https://github.com/nnstreamer/nntrainer
 * @author Daniel Jang <minhyukjang@snu.ac.kr>
 * @date   
 */

# include <bottleneck_layer.h>

# include <tensor.h>
// # include <cassert>
# include <iostream>

namespace custom {
 size_t constexpr static MU_IDX = 0;
 size_t constexpr static LOGVAR_IDX = 1;
 size_t constexpr static OUTPUT_IDX = 0;
 void BottleneckLayer :: finalize(nntrainer :: InitLayerContext & context) {
  NNTR_THROW_IF(context.getNumInputs() != 2, std :: invalid_argument);
  sigmaepsilon_idx = context.requestTensor({100, 1, 1, 16}, "sigmaepsilon");
  context.setOutputDimensions({16});
 };
 void BottleneckLayer :: forwarding(nntrainer :: RunLayerContext & context, bool training) {
  nntrainer :: Tensor const & mu = context.getInput(MU_IDX);
  nntrainer :: Tensor const & logvar = context.getInput(LOGVAR_IDX);
  nntrainer :: Tensor & sigmaepsilon = context.getTensor(sigmaepsilon_idx);
  nntrainer :: Tensor & output = context.getOutput(OUTPUT_IDX);
  // std :: cerr << mu.getDim() << std :: endl;
  // std :: cerr << logvar.getDim() << std :: endl;
  // std :: cerr << output.getDim() << std :: endl;
  nntrainer :: Tensor sigma = logvar.multiply(0.5);
  sigma.apply_i<float>([] (float logsigma) {
   return exp(logsigma);
  });
  sigmaepsilon.setRandNormal(0, 1);
  sigmaepsilon.multiply_i(sigma);
  output = nntrainer :: Tensor {sigmaepsilon}; // Copied?
  output.add_i(mu);
 }
 void BottleneckLayer :: calcDerivative(nntrainer :: RunLayerContext & context) {
  nntrainer :: Tensor & dJdmu = context.getOutgoingDerivative(MU_IDX);
  nntrainer :: Tensor & dJdlogvar = context.getOutgoingDerivative(LOGVAR_IDX);
  nntrainer :: Tensor const & sigmaepsilon = context.getTensor(sigmaepsilon_idx);
  nntrainer :: Tensor const & dJdz = context.getIncomingDerivative(OUTPUT_IDX);
  // std :: cout << "AA\n" << dJdmu << dJdlogvar << sigmaepsilon << dJdz << std :: endl;
  dJdmu = nntrainer :: Tensor {dJdz}; // Copied?
  dJdlogvar = sigmaepsilon.multiply(dJdz);
  dJdlogvar.divide_i(2);
 }
}

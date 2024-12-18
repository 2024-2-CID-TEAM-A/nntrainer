// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2024 Daniel Jang <minhyukjang@snu.ac.kr>
 *
 * @file   vae_loss_layer.cpp
 * @brief  
 * @see    https://github.com/nnstreamer/nntrainer
 * @author Daniel Jang <minhyukjang@snu.ac.kr>
 * @date   
 */

# include <vae_loss_layer.h>
# include <iostream>


namespace custom {
 float constexpr static MSEweight = 1;
 float constexpr static KLDweight = 3;

 size_t constexpr static RECON_X_IDX = 0;
 size_t constexpr static MU_IDX = 1;
 size_t constexpr static LOGVAR_IDX = 2;
 size_t constexpr static LABEL_IDX = 0;
 size_t constexpr static OUTPUT_IDX = 0;
 void VAELossLayer :: finalize(nntrainer :: InitLayerContext & context) {
  NNTR_THROW_IF(context.getNumInputs() != 3, std :: invalid_argument);
  context.setOutputDimensions({context.getInputDimensions()[0]});
 };
 void VAELossLayer :: forwarding(nntrainer :: RunLayerContext & context, bool training) {
  nntrainer :: Tensor const & recon_x = context.getInput(RECON_X_IDX);
  nntrainer :: Tensor const & mu = context.getInput(MU_IDX);
  nntrainer :: Tensor const & logvar = context.getInput(LOGVAR_IDX);
  nntrainer :: Tensor const & label = context.getLabel(LABEL_IDX);
  nntrainer :: Tensor & output = context.getOutput(OUTPUT_IDX);
  // std :: cerr << recon_x.getDim() << std :: endl;
  // std :: cerr << mu.getDim() << std :: endl;
  // std :: cerr << logvar.getDim() << std :: endl;
  // std :: cerr << label.getDim() << std :: endl;
  // std :: cerr << output.getDim() << std :: endl;
  // output = nntrainer :: Tensor {recon_x}; // 通過!
  output.fill(recon_x);
  l = logvar.add(1);
  l.subtract_i(mu.pow(2));
  l.subtract_i(logvar.apply<float>([] (float logvar) {
   return exp(logvar);
  }));
  l.divide_i(-2);
  l.multiply_i(KLDweight);
  l.add_i(recon_x.subtract(label).multiply_i(2 * MSEweight));
  LossLayer :: updateLoss(context, l);
 }
 void VAELossLayer :: calcDerivative(nntrainer :: RunLayerContext & context) {
  nntrainer :: Tensor & dJdrecon_x = context.getOutgoingDerivative(RECON_X_IDX);
  nntrainer :: Tensor & dJdmu = context.getOutgoingDerivative(MU_IDX);
  nntrainer :: Tensor & dJdlogvar = context.getOutgoingDerivative(LOGVAR_IDX);
  // nntrainer :: Tensor const & dJdJ = context.getIncomingDerivative(OUTPUT_IDX);
  nntrainer :: Tensor const & recon_x = context.getInput(RECON_X_IDX);
  nntrainer :: Tensor const & mu = context.getInput(MU_IDX);
  nntrainer :: Tensor const & logvar = context.getInput(LOGVAR_IDX);
  nntrainer :: Tensor const & label = context.getIncomingDerivative(LABEL_IDX); // 
  nntrainer :: Tensor & output = context.getOutput(OUTPUT_IDX);
  float constexpr dJdMSE = MSEweight;
  float constexpr dJdKLD = KLDweight;
  dJdrecon_x = recon_x.subtract(label);
  dJdrecon_x.multiply_i(2);
  dJdrecon_x.multiply_i(dJdMSE);
  // std :: cout << dJdrecon_x << std :: endl;
  nntrainer :: Tensor dJdb {mu.getDim()}; // or `logvar.getDim()`
  // std :: cerr << dJdb.getDim() << std :: endl;
  dJdb.setValue(- dJdKLD / 2);
  dJdmu = dJdb.multiply(-2);
  dJdmu.multiply_i(mu);
  // std :: cout << dJdmu << std :: endl;
  nntrainer :: Tensor dbdlogvar = logvar.apply<float>([] (float logvar) {
   return 1 - exp(logvar);
  });
  dJdlogvar = dJdb.multiply(dbdlogvar);
  // std :: cout << dJdlogvar << std :: endl;
 }
};

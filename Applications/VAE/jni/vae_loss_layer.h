// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2024 Daniel Jang <minhyukjang@snu.ac.kr>
 *
 * @file   vae_loss_layer.h
 * @brief  
 * @see    https://github.com/nnstreamer/nntrainer
 * @author Daniel Jang <minhyukjang@snu.ac.kr>
 * @date   
 */

# ifndef __VAE_LOSS_LAYER_H__
# define __VAE_LOSS_LAYER_H__

# include <layer_devel.h>
# include <layer_context.h>

# include <loss_layer.h>

namespace custom {
 class VAELossLayer: public nntrainer :: LossLayer {
  public:
  VAELossLayer(): LossLayer() {};
  ~ VAELossLayer() = default;
  void finalize(nntrainer :: InitLayerContext &) override;
  void forwarding(nntrainer :: RunLayerContext &, bool) override;
  void calcDerivative(nntrainer :: RunLayerContext &) override;
  void setProperty(const std :: vector<std :: string> &) override {};
  // bool requireLabel() const override { return true; };
  // bool supportBackwarding() const override { return true; };
  std :: string const getType() const override { return VAELossLayer :: type; };
  std :: string const static inline type = "vae_loss";
 };
};

# endif

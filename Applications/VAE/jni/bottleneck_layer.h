// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2024 Daniel Jang <minhyukjang@snu.ac.kr>
 *
 * @file   upsample_layer.h
 * @brief  
 * @see    https://github.com/nnstreamer/nntrainer
 * @author Daniel Jang <minhyukjang@snu.ac.kr>
 * @date   
 */

# ifndef __BOTTLENECK_LAYER_H__
# define __BOTTLENECK_LAYER_H__

# include <layer_devel.h>
# include <layer_context.h>

namespace custom {
 class BottleneckLayer: public nntrainer :: Layer {
  public:
  BottleneckLayer(): Layer() {};
  ~ BottleneckLayer() = default;
  void finalize(nntrainer :: InitLayerContext &) override;
  void forwarding(nntrainer :: RunLayerContext &, bool) override;
  void calcDerivative(nntrainer :: RunLayerContext &) override;
  bool supportBackwarding() const override { return true; };
  void setProperty(const std :: vector<std :: string> &) override {};
  std :: string const getType() const override { return BottleneckLayer :: type; };
  std :: string const static inline type = "bottleneck";
  size_t sigmaepsilon_idx;
 };
};

# endif

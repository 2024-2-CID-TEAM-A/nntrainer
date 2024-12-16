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
 class BottleneckLayer: nntrainer :: Layer {
  void finalize(nntrainer :: InitLayerContext &) override;
  void forwarding(nntrainer :: RunLayerContext &, bool) override;
  void calcDerivative(nntrainer :: RunLayerContext &) override;
  std :: string const static inline type = "bottleneck";
  size_t sigmaepsilon_idx;
 };
};

# endif

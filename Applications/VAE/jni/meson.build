VAE_sources = [
 'main.cpp',
 'bottleneck_layer.cpp',
 'vae_loss_layer.cpp',
]

VAE_dependencies = [
 nntrainer_ccapi_dep,
]

e = executable(
 'nntrainer_VAE',
 VAE_sources,
 dependencies: VAE_dependencies,
 install: get_option('install-app'),
 install_dir: application_install_dir,
)

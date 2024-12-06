// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2024 Daniel Jang <minhyukjang@snu.ac.kr>
 *
 * @file   main.cpp
 * @see    https://github.com/nnstreamer/nntrainer
 * @author Daniel Jang <minhyukjang@snu.ac.kr>
 * @author Jijoong Moon <jijoong.moon@samsung.com>
 * @date   
 * @brief  
 */

# include <sstream>

# include <memory>
# include <model.h>
# include <iostream>
# include <fstream>
# include <random>

constexpr unsigned int SEED = 0;

/**
 * @brief UserData which stores information used to feed data from data callback
 *
 */
class DataInformation {
public:
  /**
   * @brief Construct a new Data Information object
   *
   * @param num_samples number of data
   * @param filename file name to read from
   */
  DataInformation(unsigned int num_samples, const std::string &filename);
  unsigned int count;
  unsigned int num_samples;
  std::ifstream file;
  std::vector<unsigned int> idxes;
  std::mt19937 rng;
};

DataInformation::DataInformation(unsigned int num_samples,
                                 const std::string &filename) :
  count(0),
  num_samples(num_samples),
  file(filename, std::ios::in | std::ios::binary),
  idxes(num_samples) {
  std::iota(idxes.begin(), idxes.end(), 0);
  rng.seed(SEED);
  std::shuffle(idxes.begin(), idxes.end(), rng);
  if (!file.good()) {
    throw std::invalid_argument("given file is not good, filename: " +
                                filename);
  }
}

const unsigned int feature_size = 784;
const unsigned int total_label_size = 10;

/**
 * @brief     load data at specific position of file
 * @param[in] F  ifstream (input file)
 * @param[out] input input
 * @param[out] label label
 * @param[in] id th data to get
 * @retval true/false false : end of data
 */
bool getData(std::ifstream &F, float *input, float *label, unsigned int id) {
  F.clear();
  F.seekg(0, std::ios_base::end);
  uint64_t file_length = F.tellg();
  uint64_t position = (uint64_t)((feature_size + total_label_size) *
                                 (uint64_t)id * sizeof(float));

  if (position > file_length) {
    return false;
  }
  F.seekg(position, std::ios::beg);
  F.read((char *)input, sizeof(float) * feature_size);
  F.read((char *)label, sizeof(float) * total_label_size);

  return true;
}

/**
 * @brief      get data which size is batch for train
 * @param[out] outInput input vectors
 * @param[out] outLabel label vectors
 * @param[out] last if the data is finished
 * @param[in] user_data private data for the callback
 * @retval status for handling error
 */
int getSample(float **outVec, float **outLabel, bool *last, void *user_data) {
  auto data = reinterpret_cast<DataInformation *>(user_data);

  getData(data->file, *outVec, *outLabel, data->idxes.at(data->count));
  data->count++;
  if (data->count < data->num_samples) {
    *last = false;
  } else {
    *last = true;
    data->count = 0;
    std::shuffle(data->idxes.begin(), data->idxes.end(), data->rng);
  }

  return 0;
}

const unsigned int total_train_data_size = 100;
const unsigned int total_val_data_size = 100;

/**
 * @brief make "key=value" from key and value
 *
 * @tparam T type of a value
 * @param key key
 * @param value value
 * @return std::string with "key=value"
 */
template <typename T>
static std::string withKey(const std::string &key, const T &value) {
  std::stringstream ss;
  ss << key << "=" << value;
  return ss.str();
}

template <typename T>
static std::string withKey(const std::string &key,
                           std::initializer_list<T> value) {
  if (std::empty(value)) {
    throw std::invalid_argument("empty data cannot be converted");
  }

  std::stringstream ss;
  ss << key << "=";

  auto iter = value.begin();
  for (; iter != value.end() - 1; ++iter) {
    ss << *iter << ',';
  }
  ss << *iter;

  return ss.str();
}

/**
 * @brief 
 */
int main(int argc, char * argv[]) {
 if (argc != 2) {
  std :: cerr << argv[0] << " mnist_trainingSet.dat" << std :: endl;
  exit(1);
 }
 std :: string filename = argv[1];

 int image_channels = 1;
 int output_channels = 4;
 std :: unique_ptr<ml :: train :: Model> model;
 model = createModel(ml :: train :: ModelType :: NEURAL_NET);
 model -> addLayer(ml :: train :: createLayer("conv2d", {
  withKey("filters", output_channels),
  withKey("kernel_size", 3),
  withKey("stride", 2),
  withKey("padding", 1),
 }));
 model -> addLayer(ml :: train :: createLayer("batch_normalization", {
  withKey("activation", "relu"),
 }));
 model -> addLayer(ml :: train :: createLayer("conv2d", {
  withKey("filters", output_channels * 2),
  withKey("kernel_size", 3),
  withKey("stride", 2),
  withKey("padding", 1),
 }));
 model -> addLayer(ml :: train :: createLayer("batch_normalization", {
 withKey("activation", "relu"),
 }));
 model -> addLayer(ml :: train :: createLayer("conv2d", {
  withKey("filters", output_channels * 4),
  withKey("kernel_size", 3),
  withKey("stride", 2),
  withKey("padding", 1),
 }));
 model -> addLayer(ml :: train :: createLayer("batch_normalization", {
  withKey("activation", "relu"),
 }));
 model -> addLayer(ml :: train :: createLayer("conv2d", {
  withKey("filters", output_channels * 8),
  withKey("kernel_size", 3),
  withKey("stride", 2),
  withKey("padding", 1),
 }));
 model -> addLayer(ml :: train :: createLayer("batch_normalization", {
  withKey("activation", "relu"),
 }));
 model -> addLayer(ml :: train :: createLayer("conv2d", {
  withKey("filters", output_channels * 16),
  withKey("kernel_size", 3),
  withKey("stride", 2),
  withKey("padding", 1),
 }));
 model -> addLayer(ml :: train :: createLayer("batch_normalization", {
  withKey("activation", "relu"),
 }));
 model -> addLayer(ml :: train :: createLayer("dropout", {
  withKey("dropout_rate", 0.8),
 }));
 model -> addLayer(ml :: train :: createLayer("flatten"));
 model -> addLayer(ml :: train :: createLayer("bottleneck")); // TODO
 model -> addLayer(ml :: train :: createLayer("fully_connected", {
  withKey("unit", 256),
  withKey("activation", "relu"),
 }));
 model -> addLayer(ml :: train :: createLayer("reshape", {
  withKey("target_shape", "64:2:2"),
 }));
 model -> addLayer(ml :: train :: createLayer("conv2dtranspose", {
  withKey("filters", output_channels * 8),
  withKey("kernel_size", 3),
  withKey("stride", 2),
  withKey("padding", 1),
  // withKey("output_padding", 1),
 }));
 model -> addLayer(ml :: train :: createLayer("batch_normalization", {
 withKey("activation", "relu"),
 }));
 model -> addLayer(ml :: train :: createLayer("conv2dtranspose", {
  withKey("filters", output_channels * 4),
  withKey("kernel_size", 3),
  withKey("stride", 2),
  withKey("padding", 1),
  // withKey("output_padding", 1),
 }));
 model -> addLayer(ml :: train :: createLayer("batch_normalization", {
  withKey("activation", "relu"),
 }));
 model -> addLayer(ml :: train :: createLayer("conv2dtranspose", {
  withKey("filters", output_channels * 2),
  withKey("kernel_size", 3),
  withKey("stride", 2),
  withKey("padding", 1),
  // withKey("output_padding", 1),
 }));
 model -> addLayer(ml :: train :: createLayer("batch_normalization", {
  withKey("activation", "relu"),
 }));
 model -> addLayer(ml :: train :: createLayer("conv2dtranspose", {
  withKey("filters", output_channels),
  withKey("kernel_size", 3),
  withKey("stride", 2),
  withKey("padding", 1),
  // withKey("output_padding", 1),
 }));
 model -> addLayer(ml :: train :: createLayer("batch_normalization", {
  withKey("activation", "relu"),
 }));
 model -> addLayer(ml :: train :: createLayer("conv2dtranspose", {
  withKey("filters", image_channels),
  withKey("kernel_size", 3),
  withKey("stride", 2),
  withKey("padding", 1),
  // withKey("output_padding", 1),
 }));
 model -> addLayer(ml :: train :: createLayer("batch_normalization", {
  withKey("activation", "sigmoid"),
 }));

 model -> addLayer(ml :: train :: createLayer("vae_loss")); // TODO

 auto optimizer = ml :: train :: createOptimizer("adam");
 auto learningratescheduler = ml :: train :: createLearningRateScheduler("exponential");
 optimizer -> setLearningRateScheduler(std :: move(learningratescheduler));
 model -> setOptimizer(std :: move(optimizer));

 model -> compile();
 model -> initialize();

 std :: unique_ptr<DataInformation> train_user_data = std :: make_unique<DataInformation>(total_train_data_size, filename);
 std :: unique_ptr<DataInformation> valid_user_data = std :: make_unique<DataInformation>(total_val_data_size, filename);
 std :: shared_ptr<ml :: train :: Dataset> dataset_train = createDataset(ml :: train :: DatasetType :: GENERATOR, getSample, train_user_data.get());
 std :: shared_ptr<ml :: train :: Dataset> dataset_val = createDataset(ml :: train :: DatasetType :: GENERATOR, getSample, train_user_data.get());

 model -> setDataset(ml :: train :: DatasetModeType :: MODE_TRAIN, dataset_train);
 model -> setDataset(ml :: train :: DatasetModeType :: MODE_VALID, dataset_val);
 model -> train();

 // Also TODO:
 //  Use the full MNIST database
 //  Error handling
}

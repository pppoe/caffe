// Copyright 2014 BVLC and contributors.

#include <iostream> 
#include <string>
#include <vector>
#include <fstream>
#include <opencv2/opencv.hpp>

#include <google/protobuf/text_format.h>

#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/net.hpp"
#include "caffe/vision_layers.hpp"
#include "caffe/proto/caffe.pb.h"
#include "caffe/util/io.hpp"

using namespace caffe;  // NOLINT(build/namespaces)


int main(int argc, char** argv) {

  typedef float Dtype;

  const int num_required_args = 5;
  if (argc < num_required_args) {
    //LOG(ERROR)<< "feature_saved_check_point feature_extraction_proto_file list-of-images list-of-features" << std::endl;
    LOG(ERROR)<< "feature_saved_check_point feature_extraction_proto_file image-path1 image-path2" << std::endl;
    return 1;
  }

  const string param_feature_blob_name("fc4");

  int arg_pos = 1;
  const string arg_saved_net_p(argv[arg_pos++]);
  const string arg_proto_file(argv[arg_pos++]);
  const string arg_image1_p(argv[arg_pos++]);
  const string arg_image2_p(argv[arg_pos++]);

  Caffe::set_mode(Caffe::CPU);

  string pretrained_binary_proto(arg_saved_net_p);
  string feature_extraction_proto(arg_proto_file);

  shared_ptr<Net<Dtype> > feature_extraction_net(new Net<Dtype>(feature_extraction_proto, caffe::TEST));
  feature_extraction_net->CopyTrainedLayersFrom(pretrained_binary_proto);

  CHECK(feature_extraction_net->has_blob(param_feature_blob_name)) << "Unknown feature blob name " << param_feature_blob_name << " in the network " << feature_extraction_proto;

  vector<string> image_paths;
  image_paths.push_back(arg_image1_p);
  image_paths.push_back(arg_image2_p);

  const vector<Blob<Dtype>*> input_blobs = feature_extraction_net->input_blobs();
  for (size_t img_idx = 0; img_idx < image_paths.size(); img_idx++) {
      vector<Blob<Dtype>*> blobs(1);
      blobs[0] = new Blob<Dtype>();
      blobs[0]->ReshapeLike(*(input_blobs[0]));
      cv::Mat image = cv::imread(image_paths[0], CV_LOAD_IMAGE_COLOR);
      ReadImageToBlob(image, blobs[0]);

      feature_extraction_net->Forward(blobs);

      const shared_ptr<Blob<Dtype> > feature_blob = feature_extraction_net->blob_by_name(param_feature_blob_name);
      const size_t feature_length = feature_blob->count();
      std::cout << "feature_length: " << feature_length << std::endl;
      for (size_t o = 0; o < blobs.size(); o++) {
          const float *p_feature = feature_blob->cpu_data() + feature_blob->offset(o);
          for (size_t f = 0; f < feature_length; f++) {
              std::cout << p_feature[f] << " ";
          }
          std::cout << std::endl;
      }
      delete blobs[0];
  }

  //std::vector<Blob<float>*> input_vec;
  //std::vector<int> image_indices(num_features, 0);
  //for (int batch_index = 0; batch_index < num_mini_batches; ++batch_index) {
    //feature_extraction_net->Forward(input_vec);
    //for (int i = 0; i < num_features; ++i) {
      //const shared_ptr<Blob<Dtype> > feature_blob = feature_extraction_net
          //->blob_by_name(blob_names[i]);
      //int batch_size = feature_blob->num();
      //int dim_features = feature_blob->count() / batch_size;
      //const Dtype* feature_blob_data;
      //for (int n = 0; n < batch_size; ++n) {
        //datum.set_height(feature_blob->height());
        //datum.set_width(feature_blob->width());
        //datum.set_channels(feature_blob->channels());
        //datum.clear_data();
        //datum.clear_float_data();
        //feature_blob_data = feature_blob->cpu_data() +
            //feature_blob->offset(n);
        //for (int d = 0; d < dim_features; ++d) {
          //datum.add_float_data(feature_blob_data[d]);
        //}
        //int length = snprintf(key_str, kMaxKeyStrLength, "%d",
            //image_indices[i]);
        //string out;
        //CHECK(datum.SerializeToString(&out));
        //txns.at(i)->Put(std::string(key_str, length), out);
        //++image_indices[i];
        //if (image_indices[i] % 1000 == 0) {
          //txns.at(i)->Commit();
          //txns.at(i).reset(feature_dbs.at(i)->NewTransaction());
          //LOG(ERROR)<< "Extracted features of " << image_indices[i] <<
              //" query images for feature blob " << blob_names[i];
        //}
      //}  // for (int n = 0; n < batch_size; ++n)
    //}  // for (int i = 0; i < num_features; ++i)
  //}  // for (int batch_index = 0; batch_index < num_mini_batches; ++batch_index)

  return 0;
}


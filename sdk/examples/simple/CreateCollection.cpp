
#include <Status.h>
#include <Field.h>
#include <MilvusApi.h>
#include <interface/ConnectionImpl.h>
#include "utils/Utils.h"

int main(int argc , char**argv) {
  
  TestParameters parameters = milvus_sdk::Utils::ParseTestParameters(argc, argv);
  if (!parameters.is_valid){
    return 0;
  }
  auto client = milvus::ConnectionImpl();
  milvus::ConnectParam connect_param;
  connect_param.ip_address = parameters.address_.empty() ? "127.0.0.1":parameters.address_;
  connect_param.port = parameters.port_.empty() ? "19530":parameters.port_ ;
  client.Connect(connect_param);

  milvus::Status stat;
  const std::string collectin_name = "collection1";

  // Create
  milvus::FieldPtr field_ptr1 = std::make_shared<milvus::Field>();
  milvus::FieldPtr field_ptr2 = std::make_shared<milvus::Field>();
  milvus::FieldPtr field_ptr3 = std::make_shared<milvus::Field>();
  milvus::FieldPtr field_ptr4 = std::make_shared<milvus::Field>();

  field_ptr1->field_name = "field_1";
  field_ptr1->field_type = milvus::DataType::INT64;

  field_ptr2->field_name = "field_2";
  field_ptr2->field_type = milvus::DataType::FLOAT;

  field_ptr3->field_name = "field_3";
  field_ptr3->field_type = milvus::DataType::INT32;

  field_ptr4->field_name = "field_vec";
  field_ptr4->field_type = milvus::DataType::VECTOR_FLOAT;

  milvus::Mapping mapping = {collectin_name, {field_ptr1, field_ptr2, field_ptr3, field_ptr4}};

  stat = client.CreateCollection(mapping, "test_extra_params");


  // Get Collection info
  milvus::Mapping map;
  client.GetCollectionInfo(collectin_name, map);
  for (auto &f : map.fields) {
    std::cout << f->field_name << ":" << int(f->field_type) << ":" << f->dim << "DIM" << std::endl;
  }
}
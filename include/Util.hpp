#ifndef UTIL_H
#define UTIL_H

#include <vector>
#include <string>
#include <sstream>

#include "DlSystem/ITensorFactory.hpp"
#include "DlSystem/TensorShape.hpp"

// 将一个字符串根据指定的分割符, 分割成一个容器
template <typename Container> Container& split(Container& result, const typename Container::value_type & s, typename Container::value_type::value_type delimiter )
{
  // 清理结果容器 result
  result.clear();
  // 用字符串s 构造一个输入字符串流 ss
  std::istringstream ss( s );
  // 循环读取 ss 知道结束
  while (!ss.eof())
  {
    typename Container::value_type field;
    // getline可以根据分隔符拆分字符串
    getline( ss, field, delimiter );
    // 判断空字段并跳过
    if (field.empty()) continue;
    result.push_back( field );
  }
  return result;
}

// 计算张量的大小, dims 指向维度数组的指针, 张量的维度, 张量中每个元素的大小
size_t calcSizeFromDims(const zdl::DlSystem::Dimension *dims, size_t rank, size_t elementSize);

// 从文件中加载浮点数据
std::vector<float> loadFloatDataFile(const std::string& inputFile);

// 从文件中加载字节数据
std::vector<unsigned char> loadByteDataFile(const std::string& inputFile);

// 从文件中加载字节数据到 std::vector 容器中. 该函数接受两个参数, inputFile: 输入文件路径, loadVector: std::vector 容器，用于存储字节数据
template<typename T> 
bool loadByteDataFile(const std::string& inputFile, std::vector<T>& loadVector);

// 从文件中以批量方式加载字节数据
std::vector<unsigned char> loadByteDataFileBatched(const std::string& inputFile);

// 上边的函数loadByteDataFileBatched会调用, 将 loadVector 作为返回值返回 
template<typename T> 
bool loadByteDataFileBatched(const std::string& inputFile, std::vector<T>& loadVector, size_t offset);

// 从文件中加载TF8格式的数据
bool loadByteDataFileBatchedTf8(const std::string& inputFile, std::vector<uint8_t>& loadVector, size_t offset);
// 从文件中加载TF-N格式的数据
bool loadByteDataFileBatchedTfN(const std::string& inputFile, std::vector<uint8_t>& loadVector, size_t offset,
                                unsigned char& stepEquivalentTo0, float& quantizedStepSize, bool staticQuantization, int bitWidth);

// 将 ITensor 分批保存到文件中
bool SaveITensorBatched(const std::string& path, const zdl::DlSystem::ITensor* tensor, size_t batchIndex=0, size_t batchChunk=0);
// 将用户缓冲区的内容分批保存到文件中
bool SaveUserBufferBatched(const std::string& path, const std::vector<uint8_t>& buffer, size_t batchIndex=0, size_t batchChunk=0);
// 确保指定的目录存在
bool EnsureDirectory(const std::string& dir);

// 将 uint8_t 或者 uint16_t 类型的量化数据转换为浮点数据
void TfNToFloat(float *out, uint8_t *in, const unsigned char stepEquivalentTo0, const float quantizedStepSize, size_t numElement, int bitWidth);
// 浮点数数组转换为量化数组
bool FloatToTfN(uint8_t* out, unsigned char& stepEquivalentTo0, float& quantizedStepSize, bool staticQuantization, float* in, size_t numElement, int bitWidth);

// 设置可调整的维度
void setResizableDim(size_t resizableDim);
// 返回 resizable_dim 变量的值
size_t getResizableDim();

#endif

#include <fstream>
#include <iostream>
#include <vector>
#include <sstream>
#include <cstring>
#include <string>
#include <cstdlib>
#include <sys/stat.h>
#include <cerrno>
#include <limits>
#include <cmath>

#include "Util.hpp"

#include "DlSystem/ITensorFactory.hpp"
#include "DlSystem/TensorShape.hpp"

size_t resizable_dim;

size_t calcSizeFromDims(const zdl::DlSystem::Dimension *dims, size_t rank, size_t elementSize )
{
   if (rank == 0) return 0;
   size_t size = elementSize;
   while (rank--) {
      (*dims == 0) ? size *= resizable_dim : size *= *dims;
      dims++;
   }
   return size;
}

bool EnsureDirectory(const std::string& dir)
{
   auto i = dir.find_last_of('/');
   std::string prefix = dir.substr(0, i);

   // 检查路径是否为空、"." 或 ".."。如果是，则直接返回 true。
   if (dir.empty() || dir == "." || dir == "..")
   {
      return true;
   }

   // 检查路径是否存在, 如果不存在, 则递归调用 EnsureDirectory() 函数它
   if (i != std::string::npos && !EnsureDirectory(prefix))
   {
      return false;
   }

   // 尝试创建目录. 如果成功, 则返回 true
   int rc = mkdir(dir.c_str(),  S_IRWXU | S_IRGRP | S_IXGRP | S_IROTH | S_IXOTH);
   if (rc == -1 && errno != EEXIST)
   {
      return false;
   }
   else
   {
      struct stat st;
      if (stat(dir.c_str(), &st) == -1)
      {
         return false;
      }

      return S_ISDIR(st.st_mode);
   }
}

std::vector<float> loadFloatDataFile(const std::string& inputFile)
{
    std::vector<float> vec;
    loadByteDataFile(inputFile, vec);
    return vec;
}

std::vector<unsigned char> loadByteDataFile(const std::string& inputFile)
{
   std::vector<unsigned char> vec;
   loadByteDataFile(inputFile, vec);
   return vec;
}

// 将 inputFile 里的数据加载到 loadVector 变量里头
template<typename T>
bool loadByteDataFile(const std::string& inputFile, std::vector<T>& loadVector)
{
   // 打开输入文件, 检查是否可以被打开, 如果无法被打开, 函数会打印错误信息并返回 false
   std::ifstream in(inputFile, std::ifstream::binary);
   if (!in.is_open() || !in.good())
   {
      std::cerr << "Failed to open input file: " << inputFile << "\n";
   }

   // 获取输入文件的长度
   in.seekg(0, in.end);
   size_t length = in.tellg();
   in.seekg(0, in.beg);

   // 检查输入文件的长度是否是数据类型 T 的倍数。如果不是，该函数会打印错误信息并返回 false
   if (length % sizeof(T) != 0) {
      std::cerr << "Size of input file should be divisible by sizeof(dtype).\n";
      return false;
   }

   // 调整 loadVector 容器的大小，使其与输入文件的长度相同
   if (loadVector.size() == 0) {
      loadVector.resize(length / sizeof(T));
   } else if (loadVector.size() < length / sizeof(T)) {
      std::cerr << "Vector is not large enough to hold data of input file: " << inputFile << "\n";
      loadVector.resize(length / sizeof(T));
   }

   // 将输入文件中的字节数据读取到 loadVector 容器中
   if (!in.read(reinterpret_cast<char*>(&loadVector[0]), length))
   {
      std::cerr << "Failed to read the contents of: " << inputFile << "\n";
   }
   return true;
}

std::vector<unsigned char> loadByteDataFileBatched(const std::string& inputFile)
{
   std::vector<unsigned char> vec;
   size_t offset=0;
   loadByteDataFileBatched(inputFile, vec, offset);
   return vec;
}

// 从文件中批量加载字节数据到 loadVector 容器中
template<typename T>
bool loadByteDataFileBatched(const std::string& inputFile, std::vector<T>& loadVector, size_t offset)
{
    std::ifstream in(inputFile, std::ifstream::binary);
    if (!in.is_open() || !in.good())
    {
        std::cerr << "Failed to open input file: " << inputFile << "\n";
    }

    in.seekg(0, in.end);
    size_t length = in.tellg();
    in.seekg(0, in.beg);

    if (length % sizeof(T) != 0) {
        std::cerr << "Size of input file should be divisible by sizeof(dtype).\n";
        return false;
    }

    if (loadVector.size() == 0) {
        loadVector.resize(length / sizeof(T));
    } else if (loadVector.size() < length / sizeof(T)) {
        std::cerr << "Vector is not large enough to hold data of input file: " << inputFile << "\n";
    }

    loadVector.resize( (offset+1) * length / sizeof(T) );

    if (!in.read( reinterpret_cast<char*> (&loadVector[offset * length/ sizeof(T) ]), length) )
    {
        std::cerr << "Failed to read the contents of: " << inputFile << "\n";
    }
    return true;
}

/*
    out 指向存储转换后浮点数据的数组
    in 指向存储量化数据的数组
    stepEquivalentTo0 量化步长为 0 的值
    quantizedStepSize: 量化步长
    numElement: 量化数据的数量
    bitWidth: 量化数据的位宽
*/ 
void TfNToFloat(float *out,
                uint8_t *in,
                const unsigned char stepEquivalentTo0,
                const float quantizedStepSize,
                size_t numElement,
                int bitWidth)
{
   // 函数定义了一个循环, 从 0 到 numElement - 1, 依次遍历量化数据
   for (size_t i = 0; i < numElement; ++i) {
       // 如果 bitWidth 等于8
       if (8 == bitWidth) {
           // 则函数将 in[i] -> double, 并且将其减 stepEquivalentTo0, 然后乘以 quantizedStepSize, 最后将结果赋值给 out[i]
           double quantizedValue = static_cast <double> (in[i]);
           double stepEqTo0 = static_cast <double> (stepEquivalentTo0);
           out[i] = static_cast <double> ((quantizedValue - stepEqTo0) * quantizedStepSize);
       }
       // 如果 bitWidth 等于16
       else if (16 == bitWidth) {
           uint16_t *temp = (uint16_t *)in;
           // 则将 (uint16_t *)in 转换为 double 型 
           double quantizedValue = static_cast <double> (temp[i]);
           double stepEqTo0 = static_cast <double> (stepEquivalentTo0);
           out[i] = static_cast <double> ((quantizedValue - stepEqTo0) * quantizedStepSize);
       }
   }
}

/*
    out: 指向输出数组的指针，将存储量化后的值
    stepEquivalentTo0: 指向一个无符号字符变量的指针，将存储最接近 0 的步长值
    quantizedStepSize: 指向一个浮点变量的指针，将存储量化步长值
    staticQuantization: 一个布尔标志，指示量化是静态的还是动态的
    in: 指向输入数组的指针，包含浮点值
    numElement: 输入数组的元素个数
    bitWidth: 输出数组的位宽
*/
bool FloatToTfN(uint8_t* out,
                unsigned char& stepEquivalentTo0,
                float& quantizedStepSize,
                bool staticQuantization,
                float* in,
                size_t numElement,
                int bitWidth)
{
   // 储编码最小值、编码最大值、编码范围和真位宽最大值
   double encodingMin;
   double encodingMax;
   double encodingRange;
   double trueBitWidthMax = pow(2, bitWidth) -1;

   // 检查 staticQuantization 标志。如果标志未被设置，则代码进入 if 语句
   if (!staticQuantization) {
      // 找到输入数组中的最小值和最大值。std::numeric_limits <float>::max() 函数返回一个 float 可以表示的最大值，
      // std::numeric_limits <float>::min() 函数返回一个 float 可以表示的最小值
      float trueMin = std::numeric_limits <float>::max();
      float trueMax = std::numeric_limits <float>::min();

      for (size_t i = 0; i < numElement; ++i) {
         trueMin = fmin(trueMin, in[i]);
         trueMax = fmax(trueMax, in[i]);
      }

      double stepCloseTo0;

      if (trueMin > 0.0f) {
         stepCloseTo0 = 0.0;
         encodingMin = 0.0;
         encodingMax = trueMax;
      } else if (trueMax < 0.0f) {
         stepCloseTo0 = trueBitWidthMax;
         encodingMin = trueMin;
         encodingMax = 0.0;
      } else {
         double trueStepSize = static_cast <double>(trueMax - trueMin) / trueBitWidthMax;
         stepCloseTo0 = -trueMin / trueStepSize;
         if (stepCloseTo0 == round(stepCloseTo0)) {
            // 0.0 is exactly representable
            encodingMin = trueMin;
            encodingMax = trueMax;
         } else {
            stepCloseTo0 = round(stepCloseTo0);
            encodingMin = (0.0 - stepCloseTo0) * trueStepSize;
            encodingMax = (trueBitWidthMax - stepCloseTo0) * trueStepSize;
         }
      }

      const double minEncodingRange = 0.01;
      encodingRange = encodingMax - encodingMin;
      quantizedStepSize = encodingRange / trueBitWidthMax;
      stepEquivalentTo0 = static_cast <unsigned char> (round(stepCloseTo0));

      if (encodingRange < minEncodingRange) {
         std::cerr << "Expect the encoding range to be larger than " << minEncodingRange << "\n"
                   << "Got: " << encodingRange << "\n";
         return false;
      }
   }
   else
   {
      // 计算 encodingMin 和 encodingMax
      if (bitWidth == 8) {
        // 如果位宽为 8 位，则 encodingMin 将被设置为 (0 - 最接近 0 的步长值) 乘以量化步长
        encodingMin = (0 - static_cast <uint8_t> (stepEquivalentTo0)) * quantizedStepSize;
      } else if (bitWidth == 16) {
        // 位宽为 16 位，则 encodingMin 将被设置为 (0 - 最接近 0 的步长值的 uint16_t 表示) 乘以量化步长
        encodingMin = (0 - static_cast <uint16_t> (stepEquivalentTo0)) * quantizedStepSize;
      } else {
         std::cerr << "Quantization bitWidth is invalid " << std::endl;
         return false;
      }
      // encodingMax 将被设置为 trueBitWidthMax 减去最接近 0 的步长值，并乘以量化步长
      encodingMax = (trueBitWidthMax - stepEquivalentTo0) * quantizedStepSize;
      // encodingRange 将被设置为 encodingMax 减去 encodingMin
      encodingRange = encodingMax - encodingMin;
   }

   for (size_t i = 0; i < numElement; ++i) {
      int quantizedValue = round(trueBitWidthMax * (in[i] - encodingMin) / encodingRange);

      if (quantizedValue < 0)
         quantizedValue = 0;
      else if (quantizedValue > (int)trueBitWidthMax)
         quantizedValue = (int)trueBitWidthMax;

      if(bitWidth == 8){
         out[i] = static_cast <uint8_t> (quantizedValue);
      }
      else if(bitWidth == 16){
         uint16_t *temp = (uint16_t *)out;
         temp[i] = static_cast <uint16_t> (quantizedValue);
      }
   }
   return true;
}

bool loadByteDataFileBatchedTfN(const std::string& inputFile, std::vector<uint8_t>& loadVector, size_t offset,
                                unsigned char& stepEquivalentTo0, float& quantizedStepSize, bool staticQuantization, int bitWidth)
{
   std::ifstream in(inputFile, std::ifstream::binary);
   std::vector<float> inVector;
   if (!in.is_open() || !in.good())
   {
      std::cerr << "Failed to open input file: " << inputFile << "\n";
   }

   in.seekg(0, in.end);
   size_t length = in.tellg();
   in.seekg(0, in.beg);

   if (loadVector.size() == 0) {
      loadVector.resize(length / sizeof(uint8_t));
   } else if (loadVector.size() < length/sizeof(float)) {
      std::cerr << "Vector is not large enough to hold data of input file: " << inputFile << "\n";
   }

   inVector.resize(length / sizeof(float));
   if (!in.read( reinterpret_cast<char*> (&inVector[0]), length) )
   {
      std::cerr << "Failed to read the contents of: " << inputFile << "\n";
   }
   int elementSize = bitWidth / 8;
   size_t dataStartPos = (offset * length * elementSize) / sizeof(float);
   if(!FloatToTfN(&loadVector[dataStartPos], stepEquivalentTo0, quantizedStepSize, staticQuantization, inVector.data(), inVector.size(), bitWidth))
   {
     return false;
   }
   return true;
}

bool loadByteDataFileBatchedTf8(const std::string& inputFile, std::vector<uint8_t>& loadVector, size_t offset)
{
   std::ifstream in(inputFile, std::ifstream::binary);
   std::vector<float> inVector;
   if (!in.is_open() || !in.good())
   {
      std::cerr << "Failed to open input file: " << inputFile << "\n";
   }

   in.seekg(0, in.end);
   size_t length = in.tellg();
   in.seekg(0, in.beg);

   if (loadVector.size() == 0) {
      loadVector.resize(length / sizeof(uint8_t));
   } else if (loadVector.size() < length/sizeof(float)) {
      std::cerr << "Vector is not large enough to hold data of input file: " << inputFile << "\n";
   }

   inVector.resize((offset+1) * length / sizeof(uint8_t));
   if (!in.read( reinterpret_cast<char*> (&inVector[offset * length/ sizeof(uint8_t) ]), length) )
   {
      std::cerr << "Failed to read the contents of: " << inputFile << "\n";
   }

   unsigned char stepEquivalentTo0;
   float quantizedStepSize;
   if(!FloatToTfN(loadVector.data(), stepEquivalentTo0, quantizedStepSize, false, inVector.data(), loadVector.size(), 8))
   {
       return false;
   }
   return true;
}

bool SaveITensorBatched(const std::string& path, const zdl::DlSystem::ITensor* tensor, size_t batchIndex, size_t batchChunk)
{
   // 如果 batchChunk 为0, 则将其设置为 tensor 的大小
   if(batchChunk == 0)
      batchChunk = tensor->getSize();
   // 创建输出文件夹，如果它不存在
   auto idx = path.find_last_of('/');
   if (idx != std::string::npos)
   {
      std::string dir = path.substr(0, idx);
      if (!EnsureDirectory(dir))
      {
         std::cerr << "Failed to create output directory: " << dir << ": "
                   << std::strerror(errno) << "\n";
         return false;
      }
   }

   // 打开输出文件进行写入。
   std::ofstream os(path, std::ofstream::binary);
   if (!os)
   {
      std::cerr << "Failed to open output file for writing: " << path << "\n";
      return false;
   }

   // 将 `tensor` 中从 `batchIndex * batchChunk` 到 `(batchIndex+1) * batchChunk` 的元素写入文件。
   for ( auto it = tensor->cbegin() + batchIndex * batchChunk; it != tensor->cbegin() + (batchIndex+1) * batchChunk; ++it )
   {
      float f = *it;
      if (!os.write(reinterpret_cast<char*>(&f), sizeof(float)))
      {
         std::cerr << "Failed to write data to: " << path << "\n";
         return false;
      }
   }
   return true;
}

bool SaveUserBufferBatched(const std::string& path, const std::vector<uint8_t>& buffer, size_t batchIndex, size_t batchChunk)
{
   // 如果 batchChunk 参数是 0, 则将其设置为 buffer 的大小, 意味着整个 buffer 做一个批次处理
   if(batchChunk == 0)
      batchChunk = buffer.size();
   // Create the directory path if it does not exist
   auto idx = path.find_last_of('/');
   if (idx != std::string::npos)
   {
      std::string dir = path.substr(0, idx);
      // 确认批处理的目录是否存在, 如果不存在, 则创建
      if (!EnsureDirectory(dir))
      {
         std::cerr << "Failed to create output directory: " << dir << ": "
                   << std::strerror(errno) << "\n";
         return false;
      }
   }

   // 打开 path 文件，以二进制方式写入
   std::ofstream os(path, std::ofstream::binary);
   if (!os)
   {
      std::cerr << "Failed to open output file for writing: " << path << "\n";
      return false;
   }

   // 循环遍历 buffer 的当前批次。对于每个元素，将其写入文件
   for ( auto it = buffer.begin() + batchIndex * batchChunk; it != buffer.begin() + (batchIndex+1) * batchChunk; ++it )
   {
      uint8_t u = *it;
      if(!os.write((char*)(&u), sizeof(uint8_t)))
      {
        std::cerr << "Failed to write data to: " << path << "\n";
        return false;
      }
   }
   return true;
}

void setResizableDim(size_t resizableDim)
{
    // 将 resizableDim 的值赋值给 resizable_dim 变量
    resizable_dim = resizableDim;
}

size_t getResizableDim()
{
    return resizable_dim;
}

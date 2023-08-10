import numpy as np
import argparse
import os 

from PIL import Image

def __get_img_to_raw_np(img_filepath):
    '''
    获取图像的原始数组

    参数：
        img_filepath (src): 图像文件的路径

    返回:
        numpy.ndarray: 图像的原始数组

    Raises:
        RuntimeError: 如果图像的通道数不等于3 (不是RGB图像)，将引发运行时错误。
    '''
    img_filepath = os.path.abspath(img_filepath)
    img = Image.open(img_filepath)
    img_ndarray = np.array(img)
    # 检查是否为 RGB 图像, shape[0]: h, shape[1] w, shape[2]: c
    if img_ndarray.shape[2]!=3:
        raise RuntimeError('Require image with rgb but channel is %d' % img_ndarray.shape[2])
    return img_ndarray

def __create_raw_from_jpg(img_filepath, dst, req_bgr_raw, save_uint8):
    '''
    从 JPEG 图像创建 SNPE 所需的原始数组, 最后保存为 .raw 文件

    Args:
        img_filepath (str): JPEG图像文件的路径。
        req_bgr_raw (bool): 是否需要将图像数据转换为BGR格式。
        save_uint8 (bool): 是否将图像数据保存为8位整数类型(uint8)。

    '''
    img_raw = __get_img_to_raw_np(img_filepath)

    snpe_raw = img_raw.astype(np.float32)

    if req_bgr_raw:
        snpe_raw = snpe_raw[..., ::-1]

    if save_uint8:
        snpe_raw = snpe_raw.astype(np.uint8)
    else:
        snpe_raw = snpe_raw.astype(np.float32)

    filename, _ = os.path.splitext(os.path.basename(img_filepath))
    snpe_raw_filename = os.path.join(dst, filename + '.raw')
    snpe_raw.tofile(snpe_raw_filename)

def convert_images_to_raw(src, dst, req_bgr_raw=False, save_uint8=False):
    '''
    将图像转换为.raw格式。

    参数：
        src (str)：源图像文件夹的路径。
        dst (str)：目标图像文件夹的路径。
        req_bgr_raw (bool): 是否需要将图像数据转换为BGR格式。默认为False。
        save_uint8 (bool): 是否将图像数据保存为8位整数类型(uint8)。默认为False。

    返回：
        None

    '''
    print("Converting images to .raw")
    for root, dirs, files in os.walk(src):
        for jpgs in files:
            src_image=os.path.join(root, jpgs)
            if ('.jpg' in src_image):
                print(src_image)
                __create_raw_from_jpg(src_image, dst, req_bgr_raw, save_uint8)

def main():
    parser = argparse.ArgumentParser(description="Convert images to .raw", formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument('-s', '--src', type=str, required=True, help='Source folder for images')
    parser.add_argument('-d', '--dst', type=str, required=True, help='Destination folder for converted images')
    parser.add_argument('--req_bgr_raw', action='store_true', help='Convert image data to BGR format')
    parser.add_argument('--save_uint8', action='store_true', help='Save image data as uint8 format')

    args = parser.parse_args()

    src = os.path.abspath(args.src)
    dst = os.path.abspath(args.dst)
    req_bgr_raw = args.req_bgr_raw
    save_uint8  = args.save_uint8

    convert_images_to_raw(src, dst, req_bgr_raw, save_uint8)

if __name__ == '__main__':
    exit(main())
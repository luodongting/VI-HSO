#pragma once
#include <stdio.h>
#include <iostream>
#include <dirent.h>
#include <opencv2/opencv.hpp>
namespace hso {
class ImageReader
{
 public:
    ImageReader(std::string image_folder, cv::Size new_size);       
    int getDir(std::string dir, std::vector<std::string> &files);   
    cv::Mat readImage(int image_index);                             
    int getNumImages() { return (int)m_files.size(); }
 private:
    cv::Size m_img_new_size;   
    std::vector<std::string> m_files;  
};
}

#include "vihso/ImageReader.h"
#include <opencv2/imgcodecs/imgcodecs.hpp>
#include <opencv2/imgproc/types_c.h>
namespace hso {
ImageReader::ImageReader(std::string image_folder, cv::Size new_size)
{
    getDir(image_folder, m_files);
    printf("ImageReader: got %d files in %s!\n", (int)m_files.size(), image_folder.c_str());
    m_img_new_size = new_size;
}
cv::Mat ImageReader::readImage(int image_index)
{
    cv::Mat imgC2 = cv::imread(m_files.at(image_index));    
    cv::Mat image;
    cv::cvtColor(imgC2, image, CV_BGR2GRAY);
    if(!image.data)
    {
        std::cout << "ERROR READING IMAGE " << m_files.at(image_index) << std::endl;
        return cv::Mat();
    }
    cv::resize(image, image, m_img_new_size);
    return image;
}
int ImageReader::getDir(std::string dir, std::vector<std::string> &files)
{
    DIR *dp;
    struct dirent *dirp;
    if((dp = opendir(dir.c_str())) == NULL) 
    {
        return -1;
    }
    while ((dirp = readdir(dp)) != NULL)    
    {
        std::string name = std::string(dirp->d_name);
        if(name != "." && name != "..")
            files.push_back(name);
    }
    closedir(dp);
    std::sort(files.begin(), files.end());
    if(dir.at(dir.length() - 1) != '/')
        dir = dir+"/";
    for(unsigned int i = 0; i < files.size(); i++)
    {
        if(files[i].at(0) != '/')
            files[i] = dir + files[i];
    }
    return (int)files.size();
}
}

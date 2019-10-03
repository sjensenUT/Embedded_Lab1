#include<string>
void calcPrevCoords(int coords[9][4], int prevCoords[9][4], int stride, int filterSize, int prevWidth, int prevHeight, std::string prevLayerType);
void prevLayerCoords(int coords[4], int prevCoords[4], int stride, int filterSize, int prevWidth, int prevHeight, std::string layerType);
float* getSubArray(float arr[], const int coords[], int width, int height, int numChannels);
//float* horzCat(float arr1[], float arr2[], int width1, int width2, int height, int numChannels);
//float* vertCat(float arr1[], float arr2[], int width, int height1, int height2, int numChannels);
float* mergeTiles(float **tiles, const int widths[], const int heights[], int numChannels);

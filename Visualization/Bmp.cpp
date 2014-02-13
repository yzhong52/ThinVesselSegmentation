// Bmp.cpp
// =======
// BMP image loader
// It reads only 8/24/32-bit uncompressed and 8-bit RLE compression format.
//
// 2006-10-17: Improved flipImage()
// 2006-10-10: Added getError() to return the last error message.
// 2006-10-07: Fixed handling paddings if the width is not divisible by 4.
// 2006-09-25: Added 8-bit grayscale read and save (it is indexed mode).
//
//  AUTHOR: Song Ho Ahn (song.ahn@gmail.com)
// CREATED: 2006-05-08
// UPDATED: 2006-10-17
///////////////////////////////////////////////////////////////////////////////

#include <fstream>
#include <iostream>
#include <cstring>                      // for memcpy()
#include "Bmp.h"
using std::ifstream;
using std::ofstream;
using std::ios;
using std::cout;
using std::endl;
using namespace Image;



///////////////////////////////////////////////////////////////////////////////
// default constructor
///////////////////////////////////////////////////////////////////////////////
Bmp::Bmp() : width(0), height(0), bitCount(0), dataSize(0), data(0), dataRGB(0),
             errorMessage("No error.")
{
}



///////////////////////////////////////////////////////////////////////////////
// copy constructor
// We need DEEP COPY for dynamic memory variables because the compiler inserts
// default copy constructor automatically for you, BUT it is only SHALLOW COPY
///////////////////////////////////////////////////////////////////////////////
Bmp::Bmp(const Bmp &rhs)
{
    // copy member variables from right-hand-side object
    width = rhs.getWidth();
    height = rhs.getHeight();
    bitCount = rhs.getBitCount();
    dataSize = rhs.getDataSize();
    errorMessage = rhs.getError();

    if(rhs.getData())       // allocate memory only if the pointer is not NULL
    {
        data = new unsigned char[dataSize];
        memcpy(data, rhs.getData(), dataSize); // deep copy
    }
    else
        data = 0;           // array is not allocated yet, set to 0

    if(rhs.getDataRGB())    // allocate memory only if the pointer is not NULL
    {
        dataRGB = new unsigned char[dataSize];
        memcpy(dataRGB, rhs.getDataRGB(), dataSize); // deep copy
    }
    else
        dataRGB = 0;        // array is not allocated yet, set to 0
}



///////////////////////////////////////////////////////////////////////////////
// default destructor
///////////////////////////////////////////////////////////////////////////////
Bmp::~Bmp()
{
    // deallocate data array
    delete [] data;
    data = 0;
    delete [] dataRGB;
    dataRGB = 0;
}



///////////////////////////////////////////////////////////////////////////////
// override assignment operator
///////////////////////////////////////////////////////////////////////////////
Bmp& Bmp::operator=(const Bmp &rhs)
{
    if(this == &rhs)        // avoid self-assignment (A = A)
        return *this;

    // copy member variables
    width = rhs.getWidth();
    height = rhs.getHeight();
    bitCount = rhs.getBitCount();
    dataSize = rhs.getDataSize();
    errorMessage = rhs.getError();

    if(rhs.getData())       // allocate memory only if the pointer is not NULL
    {
        data = new unsigned char[dataSize];
        memcpy(data, rhs.getData(), dataSize);
    }
    else
        data = 0;

    if(rhs.getDataRGB())   // allocate memory only if the pointer is not NULL
    {
        dataRGB = new unsigned char[dataSize];
        memcpy(dataRGB, rhs.getDataRGB(), dataSize);
    }
    else
        dataRGB = 0;

    return *this;
}



///////////////////////////////////////////////////////////////////////////////
// clear out the exsiting values
///////////////////////////////////////////////////////////////////////////////
void Bmp::init()
{
    width = height = bitCount = dataSize = 0;
    errorMessage = "No error.";

    delete [] data;
    data = 0;
    delete [] dataRGB;
    dataRGB = 0;
}



///////////////////////////////////////////////////////////////////////////////
// print itself for debug
///////////////////////////////////////////////////////////////////////////////
void Bmp::printSelf() const
{
    cout << "===== Bmp =====\n"
         << "Width: " << width << " pixels\n"
         << "Height: " << height << " pixels\n"
         << "Bit Count: " << bitCount << " bits\n"
         << "Data Size: " << dataSize  << " bytes\n"
         << endl;
}



///////////////////////////////////////////////////////////////////////////////
// read a BMP image header infos and datafile and load
///////////////////////////////////////////////////////////////////////////////
bool Bmp::read(const char* fileName)
{
    this->init();   // clear out all values

    // check NULL pointer
    if(!fileName)
    {
        errorMessage = "File name is not defined (NULL pointer).";
        return false;
    }

    // open a BMP file as binary mode
    ifstream inFile;
    inFile.open(fileName, ios::binary);         // binary mode
    if(!inFile.good())
    {
        errorMessage = "Failed to open a BMP file to read.";
        return false;            // exit if failed
    }

    // list of entries in BMP header
    char id[2];             // magic identifier "BM" (2 bytes)
    int fileSize;           // file size in bytes (4)
    short reserved1;        // reserved 1 (2)
    short reserved2;        // reserved 2 (2)
    int dataOffset;         // starting offset of bitmap data (4)
    int infoHeaderSize;     // info header size (4)
    int width;              // image width (4)
    int height;             // image height (4)
    short planeCount;       // # of planes (2)
    short bitCount;         // # of bits per pixel (2)
    int compression;        // compression mode (4)
    int dataSizeWithPaddings; // bitmap data size with paddings in bytes (4)
    //int xResolution;        // horizontal pixels per metre (4)
    //int yResolution;        // vertical pixels per metre (4)
    //int colorCount;         // # of colours used (4)
    //int importantColorCount;// # of important colours (4)

    // read BMP header infos
    inFile.read(id, 2);                         // should be "BM"
    inFile.read((char*)&fileSize, 4);           // should be same as file size
    inFile.read((char*)&reserved1, 2);          // should be 0
    inFile.read((char*)&reserved2, 2);          // should be 0
    inFile.read((char*)&dataOffset, 4);
    inFile.read((char*)&infoHeaderSize, 4);     // should be 40
    inFile.read((char*)&width, 4);
    inFile.read((char*)&height, 4);
    inFile.read((char*)&planeCount, 2);         // should be 1
    inFile.read((char*)&bitCount, 2);           // 1, 4, 8, 24, or 32
    inFile.read((char*)&compression, 4);        // 0(uncompressed), 1(8-bit RLE), 2(4-bit RLE), 3(RGB with mask)
    inFile.read((char*)&dataSizeWithPaddings, 4);
    //inFile.read((char*)&xResolution, 4);
    //inFile.read((char*)&yResolution, 4);
    //inFile.read((char*)&colorCount, 4);
    //inFile.read((char*)&importantColorCount, 4);

    // check magic ID, "BM"
    if(id[0] != 'B' && id[1] != 'M')
    {
        // it is not BMP file, close the opened file and exit
        inFile.close();
        errorMessage = "Magic ID is invalid.";
        return false;
    }

    // it supports only 8-bit grayscale, 24-bit BGR or 32-bit BGRA
    if(bitCount < 8)
    {
        inFile.close();
        errorMessage = "Unsupported format.";
        return false;
    }

    // it supports only uncompressed and 8-bit RLE compressed format
    if(compression > 1)
    {
        inFile.close();
        errorMessage = "Unsupported compression mode.";
        return false;
    }

    // do not trust the file size in header, recalculate it
    inFile.seekg(0, ios::end);
    fileSize = inFile.tellg();

    // compute the number of paddings
    // In BMP, each scanline must be divisible evenly by 4.
    // If not divisible by 4, then each line adds
    // extra paddings. So it can be divided evenly by 4.
    int paddings = (4 - ((width * bitCount / 8) % 4)) % 4;

    // compute data size without paddings
    int dataSize = width * height * bitCount / 8;

    // recompute data size with paddings (do not trust the data size in header)
    dataSizeWithPaddings = fileSize - dataOffset;   // it maybe greater than "dataSize+(height*paddings)" because 4-byte boundary for file size

    // now it is ready to store info and image data
    this->width = width;
    this->height = height;
    this->bitCount = bitCount;
    this->dataSize = dataSize;

    // allocate data arrays
    // add extra bytes for paddings if width is not divisible by 4
    data = new unsigned char [dataSizeWithPaddings];
    dataRGB = new unsigned char [dataSize];

/*@@ we don't use palette for 8-bit indexed grayscale mode. Instead, we use the index value as the intensity of the pixel.
    // for loading palette
    unsigned char* palette = 0; // for palette for indexed mode
    int paletteSize = 0;

    // if bit count is 8 (256 grayscale), then it uses palette (indexed mode)
    // build palette lookup table = (4 * colorCount) bytes
    if(bitCount == 8)
    {
        // count palette size
        // palette is placed between BMP header and data
        paletteSize = dataOffset - 54;              // BMP header size is 54 bytes total

        // allocate palette array
        palette = new unsigned char[paletteSize];

        // get number of colors used
        int colorCount = paletteSize / 4;       // each palette has 4 entries(B,G,R,A)

        // copy palette data
        inFile.seekg(54, ios::beg);         // palette starts right after BMP header block (54 bytes)
        inFile.read((char*)palette, paletteSize);
    }
*/

    if(compression == 0)                    // uncompressed
    {
        inFile.seekg(dataOffset, ios::beg); // move cursor to the starting position of data
        inFile.read((char*)data, dataSizeWithPaddings);
    }
    else if(compression == 1)               // 8-bit RLE(Run Length Encode) compressed
    {
        // get length of encoded data
        int size = fileSize - dataOffset;

        // allocate tmp array to store the encoded data
        unsigned char *encData = new unsigned char[size];

        // read data from file
        inFile.seekg(dataOffset, ios::beg);
        inFile.read((char*)encData, size);

        // decode RLE into image data buffer
        decodeRLE8(encData, data);

        // deallocate encoded data buffer after decoding
        delete [] encData;
    }

    // close it after reading
    inFile.close();

    // we don't need paddings, trim paddings from each line
    // Note that there is no padding in RLE compressed data
    if(compression == 0 && paddings > 0)
    {
        int lineWidth = width * bitCount / 8;

        // copy line by line
        for(int i = 1; i < height; ++i)
        {
            memcpy(&data[i*lineWidth], &data[i*(lineWidth+paddings)], lineWidth);
        }
    }

    // BMP is bottom-to-top orientation by default, flip image vertically
    // But if the height is negative value, then it is top-to-bottom orientation.
    if(height > 0)
        flipImage(data, width, height, bitCount/8);

    // the colour components order of BMP image is BGR
    // convert image data to RGB order for convenience
    memcpy(dataRGB, data, dataSize);    // copy data to dataRGB first
    if(bitCount == 24 || bitCount == 32)
        swapRedBlue(dataRGB, dataSize, bitCount/8);

    return true;
}



///////////////////////////////////////////////////////////////////////////////
// save an image as an uncompressed BMP format
// We assume the source image is RGB order, so it must be converted BGR order.
///////////////////////////////////////////////////////////////////////////////
bool Bmp::save(const char* fileName, int w, int h, int channelCount, const unsigned char* data)
{
    // reset error message
    errorMessage = "No error.";

    if(!fileName || !data)
    {
        errorMessage = "File name is not specified (NULL pointer).";
        return false;
    }

    if(w == 0 || h == 0)
    {
        errorMessage = "Zero width or height.";
        return false;
    }

    // list of entries in BMP header
    char id[2];             // magic identifier "BM" (2 bytes)
    int fileSize;           // file size in bytes (4)
    short reserved1;        // reserved 1 (2)
    short reserved2;        // reserved 2 (2)
    int dataOffset;         // starting offset of bitmap data (4)
    int infoHeaderSize;     // info header size (4)
    int width;              // image width (4)
    int height;             // image height (4)
    short planeCount;       // # of planes (2)
    short bitCount;         // # of bits per pixel (2)
    int compression;        // compression mode (4)
    int dataSizeWithPaddings; // bitmap data size in bytes with padding (4)
    int xResolution;        // horizontal pixels per metre (4)
    int yResolution;        // vertical pixels per metre (4)
    int colorCount;         // # of colours used (4)
    int importantColorCount;// # of important colours (4)
    
    int paletteSize;        // size of palette block in bytes

    // compute paddings per each line
    // In BMP, each scanline must be divisible evenly by 4
    // If not, add extra paddings in each line, it can be divisible by 4.
    int paddings = (4 - ((w * channelCount) % 4)) % 4;

    // compute data size without paddings
    int dataSize = w * h * channelCount;

    // fill vars for BMP header infos
    id[0] = 'B';
    id[1] = 'M';
    reserved1 = reserved2 = 0;
    width = w;
    height = h;
    planeCount = 1;
    bitCount = channelCount * 8;
    compression = 0;
    dataSizeWithPaddings = dataSize + (h * paddings);
    xResolution = yResolution = 2835;   // 72 pixels/inch = 2835 pixels/m
    colorCount = 0;
    importantColorCount = 0;
    infoHeaderSize = 40;                // should be 40 bytes
    dataOffset = 54;                    // fileHeader(14) + infoHeader(40)
    fileSize = dataSizeWithPaddings + dataOffset;

    // 8-bit grayscale image need palette
    // correct colorCount, dataOffset and fileSize
    if(channelCount == 1)
    {
        colorCount = 256;                   // always use max number of colors for 8-bit gray scale
        paletteSize = colorCount * 4;       // BGRA for each
        dataOffset = 54 + paletteSize;      // add up palette size
        fileSize = dataSizeWithPaddings + dataOffset;   // reset file size
    }

    // allocate output data array
    unsigned char* tmpData = new unsigned char [dataSize];

    // copy image data
    memcpy(tmpData, data, dataSize);

    // flip the image upside down
    // If height is  negative, then it is bottom-top orientation (no need to flip)
    if(height > 0)
        flipImage(tmpData, width, height, channelCount);

    // convert RGB to BGR order
    if(channelCount == 3 || channelCount == 4)
        swapRedBlue(tmpData, dataSize, channelCount);

    // add paddings(0s) if the width of image is not divisible by 4
    unsigned char* dataWithPaddings = 0;
    if(paddings > 0)
    {
        // allocate an array
        // add extra bytes for paddings in case the width is not divisible by 4
        dataWithPaddings = new unsigned char [dataSizeWithPaddings];

        int lineWidth = width * channelCount;       // line width in bytes

        // copy single line at a time
        for(int i = 0; i < height; ++i)
        {
            // restore data by adding paddings
            memcpy(&dataWithPaddings[i*(lineWidth+paddings)], &tmpData[i*lineWidth], lineWidth);

            // insert 0s for paddings after copying the current line
            for(int j = 1; j <= paddings; ++j)
                dataWithPaddings[(i+1)*(lineWidth+paddings) - j] = (unsigned char)0;
        }
    }

    // open output file to write data
    ofstream outFile;
    outFile.open(fileName, ios::binary);
    if(!outFile.good())
    {
        errorMessage = "Failed to open an optput file.";
        return false;   // exit if failed
    }

    // write header
    outFile.put(id[0]);
    outFile.put(id[1]);
    outFile.write((char*)&fileSize, 4);
    outFile.write((char*)&reserved1, 2);
    outFile.write((char*)&reserved2, 2);
    outFile.write((char*)&dataOffset, 4);
    outFile.write((char*)&infoHeaderSize, 4);
    outFile.write((char*)&width, 4);
    outFile.write((char*)&height, 4);
    outFile.write((char*)&planeCount, 2);
    outFile.write((char*)&bitCount, 2);
    outFile.write((char*)&compression, 4);
    outFile.write((char*)&dataSizeWithPaddings, 4);
    outFile.write((char*)&xResolution, 4);
    outFile.write((char*)&yResolution, 4);
    outFile.write((char*)&colorCount, 4);
    outFile.write((char*)&importantColorCount, 4);

    // For 8-bit grayscale, insert palette between header block and data block
    if(bitCount == 8)
    {
        unsigned char* palette = new unsigned char[paletteSize]; // each entry has 4 bytes(B,G,R,A)
        buildGrayScalePalette(palette, paletteSize);

        // write palette to the file
        outFile.write((char*)palette, paletteSize);
        delete [] palette;
    }

    // write image data
    if(paddings == 0)
        outFile.write((char*)tmpData, dataSize);                        // without padding
    else
        outFile.write((char*)dataWithPaddings, dataSizeWithPaddings);   // with paddings

    // close the opened file
    outFile.close();

    // deallocate tmp buffer
    delete [] tmpData;
    delete [] dataWithPaddings;

    return true;
}



// static shared functions ****************************************************

///////////////////////////////////////////////////////////////////////////////
// decode 8-bit RLE data into uncompressed data
// This routine need 2 pointers: the pointer to the encoded input data and
// the pointer to the decoded output data. The length of input array is not
// necessary because the last 2 bytes of input data must be 00 and 01, which
// tells the end of data. So it can stop decoding process.
//
// BMP uses 2-value RLE scheme: the first value contains a count of the number
// of pixels in the run, and the second value contains the value of the pixel
// repeated. For example, 0x3 0xFF means 0xFF 0xFF 0xFF.
//
// If the first value is 0x00, then it is unencoded run mode and a pixel is not
// repeated any more. In unencode run mode, the second value is the the number
// of unencoded pixel values that follow. If the number of pixels is odd, then
// a 0x00 padding value also follows.
// 1st  2nd  EncodedValue  DecodedValue
// ===  ===  ============  ============
//  00   03  FF FE FD 00   FF FE FD
//  00   04  11 12 13 14   11 12 13 14
//
// The second value of unencoded run mode must be greater than and equal to 3.
// If the second value is less than 3, then it specifies special positioning
// operations and does not decode any data themselves.
// 1st  2nd  Meaning
// ===  ===  ==============================================
//  00   00  End of Scanline, Decode new data at the next line
//  00   01  End of Bitmap data, Stop decoding data here
//  00   02  Delta Offset, Move the cursor hori and vert direction
//
// Delta Offset operation requires 4-byte in size: the first and second should
// be 00 and 02, and the third byte is the number of pixels forward in the
// same scanline and the fourth byte is the number of rows to move. For
// example, 00 02 03 04 means move the cursor 3 pixels right, and 4 pixels
// upward. (Note that BMP is bottom-to-top orientation.)
///////////////////////////////////////////////////////////////////////////////
bool Bmp::decodeRLE8(const unsigned char *encData, unsigned char *outData)
{
    // check NULL pointer
    if(!encData || !outData)
        return false;

    unsigned char first, second;
    int i;
    bool stop = false;

    // start decoding, stop when it reaches at the end of decoded data
    while(!stop)
    {
        // grab 2 bytes at the current position
        first = *encData++;
        second = *encData++;

        if(first)                   // encoded run mode
        {
            for(i=0; i < first; ++i)
                *outData++ = second;
        }
        else
        {
            if(second == 1)         // reached the end of bitmap
                stop = true;        // must stop decoding

            else if(second == 2)    // delta mark
                encData += 2;       // do nothing, but move the cursor 2 more bytes

            else                    // unencoded run mode (second >= 3)
            {
                for(i=0; i < second; ++i)
                    *outData++ = *encData++;

                if(second % 2)      // if it is odd number, then there is a padding 0. ignore it
                    encData++;
            }
        }
    }

    return true;
}



///////////////////////////////////////////////////////////////////////////////
// BMP is bottom-to-top orientation. Flip the image vertically, so the image
// can be rendered from top to bottom orientation
///////////////////////////////////////////////////////////////////////////////
void Bmp::flipImage(unsigned char *data, int width, int height, int channelCount)
{
    if(!data) return;

    int lineSize = width * channelCount;
    unsigned char* tmp = new unsigned char [lineSize];
    int half = height / 2;

    int line1 = 0;                          // first line
    int line2 = (height - 1) * lineSize;    // last line

    // scan only half of height
    for(int i = 0; i < half; ++i)
    {
        // copy line by line
        memcpy(tmp, &data[line1], lineSize);
        memcpy(&data[line1], &data[line2], lineSize);
        memcpy(&data[line2], tmp, lineSize);

        // move to next line
        line1 += lineSize;
        line2 -= lineSize;
    }

    // deallocate temp arrays
    delete [] tmp;
}



///////////////////////////////////////////////////////////////////////////////
// swap the position of the 1st and 3rd color components (RGB <-> BGR)
///////////////////////////////////////////////////////////////////////////////
void Bmp::swapRedBlue(unsigned char *data, int dataSize, int channelCount)
{
    if(!data) return;
    if(channelCount < 3) return;            // must be 3 or 4
    if(dataSize % channelCount) return;     // must be divisible by the number of channels

    unsigned char tmp;
    int i;

    // swap the position of red and blue components
    for(i=0; i < dataSize; i+=channelCount)
    {
        tmp = data[i];
        data[i] = data[i+2];
        data[i+2] = tmp;
    }
}



///////////////////////////////////////////////////////////////////////////////
// compute the number of used colors in the 8-bit grayscale image
///////////////////////////////////////////////////////////////////////////////
int Bmp::getColorCount(const unsigned char* data, int dataSize)
{
    if(!data) return 0;

    const int MAX_COLOR = 256;  // max number of colors in 8-bit grayscale
    int i;
    int colorCount = 0;
    unsigned int colors[MAX_COLOR];

    // clear all to 0s
    memset((void*)colors, 0, MAX_COLOR);

    // increment at the same index
    for(i = 0; i < dataSize; ++i)
        colors[data[i]]++;

    // count backward the number of color used in this data
    colorCount = MAX_COLOR;
    for(i = 0; i < MAX_COLOR; ++i)
    {
        if(colors[i] == 0)
            colorCount--;
    }

    return colorCount;
}



///////////////////////////////////////////////////////////////////////////////
// build palette for 8-bit grayscale image
// Each component(B,G,R,A) of palette will have the same value as data value
// because it is grayscale.
///////////////////////////////////////////////////////////////////////////////
void Bmp::buildGrayScalePalette(unsigned char* palette, int paletteSize)
{
    if(!palette) return;

    // fill B, G, R, with same value and A is 0
    int i, j;
    for(i = 0, j = 0; i < paletteSize; i+=4, j++)
    {
        palette[i] = palette[i+1] = palette[i+2] = (unsigned char)j;
        palette[i+3] = (unsigned char)0;
    }
}

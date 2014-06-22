#ifndef CVPLOT_H
#define CVPLOT_H

#include <opencv2/core/core.hpp>
#include <string>
#include <vector>

// This namespace contains some utility functions for saving plots using OpenCV
namespace CVPlot
{

bool draw( const std::string& name,    // name of the output image
           std::vector<std::vector<double> >& funcs, // data y
           unsigned im_height = 200,        // image height
           unsigned im_width = 0           // image width, will be computed based on the size of mat_ys if left empty
            );

}

#endif // CVPLOT_H


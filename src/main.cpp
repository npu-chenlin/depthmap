#include "depthframes.hpp"
using namespace std;

int main(int argc,char** argv)
{
    GSLAM::ScopedTimer tm("total");
    svar.ParseMain(argc,argv);
    depthframes de(svar.GetString("dataset","/home/chenll/my_prog/slam/build-DepthMap-unknown-Default/result_gps.gmap"),
                   svar.GetString("savepath","./")+"result.ply");
    de.setMinScore(svar.GetDouble("score",0.1));
    de.setScale(svar.GetDouble("scale",0.1));
    de.calculatedepth();
    de.saveply();
    return 0;
}

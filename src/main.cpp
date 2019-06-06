#include "depthmap.hpp"
#include "MapFrame.h"
#include "MapPoint.h"
#include "MapHash.hpp"

#include <GSLAM/core/Dataset.h>
#include <opencv2/imgproc/imgproc.hpp>
#include <GSLAM/core/Timer.h>


struct PlyObject
{
    PlyObject(std::string file2save="out.ply"):_file2save(file2save){}
    ~PlyObject(){save(_file2save);}
    typedef pi::Point3f Vertex3f;
    typedef pi::Point3ub Color3b;
    typedef pi::Point3d Point3d;
    typedef pi::Point3f Point3f;
    
    std::string _file2save;
    std::vector<pi::Point3f>  vertices;
    std::vector<unsigned int> faces;
    std::vector<pi::Point3f>  normals;
    std::vector<pi::Point3ub> colors;
    std::vector<unsigned int> edges;
    
    void addPoint(Point3d pt,Color3b color=Color3b(255,255,255),pi::Point3f normal=Point3f(0,0,1))
    {
        vertices.push_back(pt);
        colors.push_back(color);
        normals.push_back(normal);
    }
    
    void addLine(Point3d first,Point3d second,Color3b color=Color3b(255,255,255),pi::Point3f normal=Point3f(0,0,1))
    {
        edges.push_back((uint32_t)vertices.size());
        edges.push_back((uint32_t)vertices.size()+1);
        addPoint(first,color,normal);
        addPoint(second,color,normal);
    }
    
    bool save(std::string filename)
    {
        if(filename.substr(filename.find_last_of('.')+1)!="ply") return false;
        std::fstream file;
        file.open(filename.c_str(),std::ios::out|std::ios::binary);
        if(!file.is_open()){
            fprintf(stderr,"\nERROR: Could not open File %s for writing!",(filename).c_str());
            return false;
        }
        
        uint32_t _verticesPerFace=3;
        bool binary=svar.GetInt("binary",0);
        
        file << "ply";
        if(binary)file << "\nformat binary_little_endian 1.0";
        else file << "\nformat ascii 1.0";
        file << "\nelement vertex " << vertices.size();
        file << "\nproperty float32 x\nproperty float32 y\nproperty float32 z";
        if(normals.size())
            file << "\nproperty float32 nx\nproperty float32 ny\nproperty float32 nz";
        if(colors.size())
            file << "\nproperty uchar red\nproperty uchar green\nproperty uchar blue";
        if(faces.size()){
            file << "\nelement face " << faces.size()/_verticesPerFace;
            file << "\nproperty list uint8 int32 vertex_indices";
        }
        if(edges.size()){
            file << "\nelement edge " << edges.size()/2;
            file << "\nproperty int vertex1\nproperty int vertex2";
        }
        file << "\nend_header";
        if(binary) file << "\n";
        
        for(unsigned int i=0;i<vertices.size();i++){
            if(binary){
                file.write((char*)(&(vertices[i])),sizeof(Vertex3f));
            }
            else file << "\n" << vertices[i].x << " " << vertices[i].y << " " << vertices[i].z;
            
            if(normals.size())
            {
                if(binary){
                    file.write((char*)(&(normals[i])),sizeof(Vertex3f));
                }
                else file << " " << normals[i].x << " " << normals[i].y << " " << normals[i].z;
            }
            if(colors.size()){
                if(binary){
                    file.write((char*)(&(colors[i])),sizeof(Color3b));
                }
                else file << " " << (int)(colors[i].x) << " " << (int)(colors[i].y) << " " << (int)(colors[i].z);
            }
        }
        for(unsigned int i=0;i<faces.size();i+=_verticesPerFace){
            if(binary){
                file.write((char*)(&_verticesPerFace),sizeof(uchar));
            }
            else file << "\n" << (int)_verticesPerFace;
            for(unsigned int j=0;j<_verticesPerFace;j++)
                if(binary){
                    unsigned int idx = faces[i+j];
                    file.write((char*)(&idx),sizeof(unsigned int));
                }
                else file << " " << (faces[i+j]);
        }
        for(unsigned int i=0;i<edges.size();i+=2){
            if(binary){
                unsigned int idx = edges[i];
                file.write((char*)(&idx),sizeof(unsigned int));
                idx = edges[i+1]; file.write((char*)(&idx),sizeof(unsigned int));
            }
            else file << "\n " << edges[i] << " " << edges[i+1];
        }
        
        file.close();
        return true;
    }
};

std::vector<FrameID> getNeighbors(GSLAM::FramePtr fr,GSLAM::MapPtr map){
    std::map<GSLAM::PointID,size_t> observes;
    fr->getObservations(observes);
    std::map<GSLAM::FrameID,size_t> neighborsCount;
    
    for(auto obs:observes){
        GSLAM::PointPtr pt=map->getPoint(obs.first);
        std::map<FrameID,size_t> ptObserves;
        if(!pt->getObservations(ptObserves)) continue;
        for(auto ptObs:ptObserves){
            neighborsCount[ptObs.first]++;
        }
    }
    
    std::vector<std::pair<size_t,GSLAM::FrameID>> sortedN;
    for(auto f:neighborsCount) sortedN.push_back(std::make_pair(f.second,f.first));
    std::sort(sortedN.begin(),sortedN.end());
    std::vector<FrameID> frames;
    for(int i=sortedN.size()-2;i>=0&&frames.size()<8;--i){
        frames.push_back(sortedN[i].second);
    }
    return frames;
}
void transform(std::array<double,9>& Rcw,GSLAM::Point3d& tcw,
               std::array<double,9>& Rlw,GSLAM::Point3d& tlw,
               std::array<double,9>& Rcl,GSLAM::Point3d& tcl){
    GSLAM::SE3 Tcw{GSLAM::SO3{Rcw.data()},GSLAM::SE3::Vec3{tcw.x,tcw.y,tcw.z}};
    GSLAM::SE3 Tlw{GSLAM::SO3{Rlw.data()},GSLAM::SE3::Vec3{tlw.x,tlw.y,tlw.z}};
    GSLAM::SE3 Tcl = Tcw*Tlw.inverse();
    Tcl.getRotation().getMatrix(Rcl.data());
    tcl = Tcl.getTranslation();
}
void addFrame(GSLAM::FramePtr fr,csfm::DepthmapEstimator& de,
              std::vector<cv::Mat>& images,
              std::vector<cv::Mat>& grays,
              std::vector<std::array<double,9> >& Rs,
              std::vector<std::array<double,9> >& Ks,
              std::vector<GSLAM::Point3d>& ts,
              cv::Mat& mask,int count){
    double  scale=svar.GetDouble("scale",0.1);
    std::string path;
    fr->call("GetImagePath",&path);
    path = "/home/chenlin/data/data/mavic-campus/images"+path.substr(path.rfind('/'));
    cv::Mat image=cv::imread(path);

    GSLAM::Camera cam=fr->getCamera();
    if(cam.CameraType()!="PinHole"||scale<1)
    {
        cam=GSLAM::Camera(fr->getCamera().estimatePinHoleCamera().getParameters());
        cam.applyScale(scale);
        static GSLAM::Undistorter undis;
        if(undis.cameraIn().info()!=fr->getCamera().info())
        {
            undis=GSLAM::Undistorter(fr->getCamera(),cam);
        }
        GSLAM::GImage undised;
        undis.undistort(image,undised);
        image=undised;
    }
    cv::Mat gray;
    cv::cvtColor(image,gray,CV_BGR2GRAY);

    std::vector<double>  p=cam.getParameters();
    std::array<double,9> K={p[2], 0    , p[4],
                            0   , p[3] , p[5],
                            0   , 0    , 1};
    std::array<double,9> R{1,0,0,0,1,0,0,0,1};
    fr->getPose().inverse().getRotation().getMatrix(R.data());
    GSLAM::Point3d t=fr->getPose().inverse().get_translation();
    if(mask.empty()) mask=cv::Mat::ones(image.rows,image.cols,CV_8UC1);

    images.push_back(image);
    grays.push_back(gray);
    Rs.push_back(R);
    Ks.push_back(K);
    ts.push_back(t);
    std::array<double,9> nR;
    GSLAM::Point3d nt;
    transform(Rs.back(),ts.back(),Rs.front(),ts.front(),nR,nt);
    for(auto i : nR){std::cout<<i<<" ";}
    std::cout<<std::endl;
    std::cout<<nt.x<<" "<<nt.y<<" "<<nt.z<<std::endl;
    de.AddView(Ks.back().data(),nR.data(),&nt.x,grays.back().data,
               mask.data,image.cols,image.rows);
    cv::imwrite(svar.GetString("savepath","./")+std::to_string(count)+"/img"+std::to_string(images.size())+".jpg",images.back());
}


std::vector<double> getMinMaxDepth(GSLAM::FramePtr fr,GSLAM::MapPtr map){
    std::map<GSLAM::PointID,size_t> observes;
    fr->getObservations(observes);
    std::vector<double> minmax={10000000000,0.1};
    for(auto obs : observes){
        GSLAM::ScopedTimer tm("computeMaxMindepth");
        GSLAM::PointPtr pt=map->getPoint(obs.first);
        double depth = (fr->getPose().inverse()*pt->getPose())[2];
        if(depth < minmax[0] && depth>0)minmax[0] = depth;
        if(depth>minmax[1]  && depth>0)minmax[1] = depth;
    }
    std::cout<<"mindepth:"<<minmax[0]<<"maxdepth:"<<minmax[1]<<std::endl;
    return minmax;
}

int main(int argc,char** argv)
{
    GSLAM::ScopedTimer tm("total");
    svar.ParseMain(argc,argv);
    //read gmap
    GSLAM::MapPtr map(new MapHash());
    if(!map->load(svar.GetString("dataset","/home/chenll/my_prog/slam/build-DepthMap-unknown-Default/result_gps.gmap")))
    {
        LOG(ERROR)<<"Failed to load map.";
        return -1;
    }

    GSLAM::FrameArray frames=map->getFrames();
    float minScore=svar.GetDouble("score",0.1);
    int count=0;
    for(GSLAM::FramePtr cur:frames){
        std::string dir = "mkdir -p "+svar.GetString("savepath","./")+std::to_string(count);
        std::system(dir.c_str());
        auto neighbors=getNeighbors(cur,map);
        auto minmax  = getMinMaxDepth(cur,map);
        csfm::DepthmapEstimator de;
        de.SetPatchMatchIterations(svar.GetInt("iters",20));
        de.SetMinPatchSD(svar.GetDouble("sd",1.));
        de.SetDepthRange(minmax[0],minmax[1],50);
        cv::Mat mask;
        std::vector<cv::Mat> images;
        std::vector<cv::Mat>  grays;
        std::vector<std::array<double,9> > Rs,Ks;
        std::vector<GSLAM::Point3d>        ts;
        addFrame(cur,de,images,grays,Rs,Ks,ts,mask,count);

        for(auto n:neighbors){
            auto ref=map->getFrame(n);
            addFrame(ref,de,images,grays,Rs,Ks,ts,mask,count);
        }

        csfm::DepthmapEstimatorResult result;
        std::string method=svar.GetString("method","sample");
        {
            GSLAM::ScopedTimer tm("computeDepth");
            if(method=="sample")
                de.ComputePatchMatchSample(&result);
            else if(method=="patch")
                de.ComputePatchMatch(&result);
            else
                de.ComputeBruteForce(&result);
        }
        for(int j=0;j<result.depth.total();j++){
            float& depth = result.depth.at<float>(j);
            if(depth<minmax[0])depth=0;
            if(depth>minmax[1])depth=0;
            if(result.score.at<float>(j)<minScore) depth=0;
        }
        cv::Mat depth=result.depth.clone();
        int mapminmax[]={50,245};
        for(int j=0;j<depth.total();j++){
            if(depth.at<float>(j)<minmax[0])continue;
            depth.at<float>(j) = mapminmax[0]+ (mapminmax[1]-mapminmax[0])*(depth.at<float>(j)-minmax[0])/(minmax[1]-minmax[0]);
        }
        cv::imwrite(svar.GetString("savepath","./")+std::to_string(count)+"/"+"depth.jpg",depth);
        csfm::DepthmapPruner pruner;
        std::vector<float>   merged_points,merged_normals;
        std::vector<unsigned char> merged_colors;
        std::vector<unsigned char> merged_labels;
        pruner.AddView(Ks[0].data(),Rs[0].data(),&ts[0].x,(float*)result.depth.data,(float*)result.plane.data,
                images[0].data,mask.data,images[0].cols,images[0].rows);
        pruner.Prune(&merged_points,&merged_normals,&merged_colors,&merged_labels);

        PlyObject ply(svar.GetString("savepath","./")+std::to_string(count)+"/"+std::to_string(count)+".ply");
        for(int i=0;i*3<merged_points.size();i++){
            ply.addPoint(GSLAM::Point3d(merged_points[3*i+0],merged_points[3*i+1],merged_points[3*i+2]),
                    GSLAM::Point3ub(merged_colors[3*i+0],merged_colors[3*i+1],merged_colors[3*i+2]),
                    GSLAM::Point3d(merged_normals[3*i+0],merged_normals[3*i+1],merged_normals[3*i+2]));
        }
        count++;
        if(count == svar.GetInt("imgnum",1)){
            break;
        }
    }
    return 0;
}
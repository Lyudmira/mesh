#include <openvdb/openvdb.h>
#include <openvdb/tools/ParticlesToLevelSet.h>
#include <openvdb/tools/VolumeToMesh.h>
#include <pcl/io/ply_io.h>
#include <pcl/point_types.h>
#include <pcl/filters/statistical_outlier_removal.h>
#include <pcl/features/normal_3d.h>
#include <pcl/search/kdtree.h>
#include <CGAL/Simple_cartesian.h>
#include <CGAL/Surface_mesh.h>
#include <CGAL/Polygon_mesh_processing/area.h>
#include <igl/writeOBJ.h>
#include <Eigen/Core>
#include <vector>
#include <string>
#include <iostream>

// Simple particle list wrapper around PCL point cloud
struct PointList {
    using value_type = openvdb::Vec3R;
    std::vector<value_type> pts;
    size_t size() const { return pts.size(); }
    void getPos(size_t i, value_type& xyz) const { xyz = pts[i]; }
};

int main(int argc, char** argv) {
    if (argc < 3) {
        std::cerr << "Usage: " << argv[0] << " input.ply output.obj" << std::endl;
        return 1;
    }
    std::string in = argv[1];
    std::string out = argv[2];

    // Load point cloud via PCL
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZ>());
    if (pcl::io::loadPLYFile(in, *cloud) != 0) {
        std::cerr << "Failed to load point cloud: " << in << std::endl;
        return 1;
    }

    // Outlier removal
    pcl::StatisticalOutlierRemoval<pcl::PointXYZ> sor;
    sor.setInputCloud(cloud);
    sor.setMeanK(8);
    sor.setStddevMulThresh(1.0);
    sor.filter(*cloud);

    // Normal estimation
    pcl::NormalEstimation<pcl::PointXYZ, pcl::Normal> ne;
    pcl::search::KdTree<pcl::PointXYZ>::Ptr tree(new pcl::search::KdTree<pcl::PointXYZ>());
    ne.setInputCloud(cloud);
    ne.setSearchMethod(tree);
    ne.setKSearch(16);
    pcl::PointCloud<pcl::Normal>::Ptr normals(new pcl::PointCloud<pcl::Normal>());
    ne.compute(*normals);

    // Convert to OpenVDB level set
    openvdb::initialize();
    using GridT = openvdb::FloatGrid;
    openvdb::math::Transform::Ptr transform = openvdb::math::Transform::createLinearTransform(0.05);
    GridT::Ptr grid = GridT::create(3.0);
    grid->setTransform(transform);

    PointList plist;
    plist.pts.reserve(cloud->size());
    for (const auto& p : cloud->points) {
        plist.pts.emplace_back(p.x, p.y, p.z);
    }
    openvdb::tools::ParticlesToLevelSet<GridT> raster(*grid);
    raster.setRmin(0.02);
    raster.rasterizeSpheres(plist);
    raster.finalize();

    // Dual contouring to mesh
    std::vector<openvdb::Vec3s> verts;
    std::vector<openvdb::Vec3I> tris;
    openvdb::tools::volumeToMesh(*grid, verts, tris);

    // Save mesh using libigl
    Eigen::MatrixXd V(verts.size(), 3);
    Eigen::MatrixXi F(tris.size(), 3);
    for (size_t i = 0; i < verts.size(); ++i) {
        V(i, 0) = verts[i].x();
        V(i, 1) = verts[i].y();
        V(i, 2) = verts[i].z();
    }
    for (size_t i = 0; i < tris.size(); ++i) {
        F(i, 0) = tris[i].x();
        F(i, 1) = tris[i].y();
        F(i, 2) = tris[i].z();
    }
    if (!igl::writeOBJ(out, V, F)) {
        std::cerr << "Failed to write OBJ: " << out << std::endl;
        return 1;
    }

    // Load mesh into CGAL and compute area for validation
    typedef CGAL::Simple_cartesian<double> K;
    typedef CGAL::Surface_mesh<K::Point_3> SurfaceMesh;
    SurfaceMesh mesh;
    for (const auto& v : verts) {
        mesh.add_vertex(K::Point_3(v.x(), v.y(), v.z()));
    }
    for (const auto& t : tris) {
        mesh.add_face(SurfaceMesh::Vertex_index(t.x()), SurfaceMesh::Vertex_index(t.y()), SurfaceMesh::Vertex_index(t.z()));
    }
    double area = CGAL::Polygon_mesh_processing::area(mesh);
    std::cout << "Pipeline complete: " << mesh.number_of_vertices() << " vertices, "
              << mesh.number_of_faces() << " faces, area=" << area << std::endl;
    return 0;
}


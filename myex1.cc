

#include <deal.II/grid/tria.h>
#include <deal.II/grid/tria_accessor.h>
#include <deal.II/grid/tria_iterator.h>
#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/tria_boundary_lib.h>
#include <deal.II/grid/grid_out.h>
#include <deal.II/grid/manifold_lib.h>
#include <deal.II/grid/manifold.h>

#include <iostream>

#include <fstream>
#include <cmath>
//This is my try in eclipse.
using namespace dealii;


void cube_grid ()
{
  Triangulation<3> triangulation;

  GridGenerator::hyper_cube (triangulation);
  triangulation.refine_global (4);

  std::ofstream out ("cube.vtk");
  GridOut grid_out;
  grid_out.write_vtk (triangulation, out);
  std::cout << "Cube mesh in 3d is written to cube.vtk" << std::endl;
}

void square_grid ()
{
    Triangulation<2> mesh;
    GridGenerator:: hyper_cube (mesh);
    mesh.refine_global(4);
    std:: ofstream out("square.eps");
    GridOut grid_out;
    grid_out.write_eps (mesh, out);
    std:: cout<<"Square Mesh in 2d is Written to square.eps"<<std::endl;
}

void L_grid_three ()
{
  Triangulation<3> triangulation;

  GridGenerator::hyper_L(triangulation, 0,2);
  triangulation.refine_global (3);

  std::ofstream out ("L_three.vtk");
  GridOut grid_out;
  grid_out.write_vtk (triangulation, out);
  std::cout << "L shape mesh in 3d is written to L.vtk" << std::endl;
}

void L_grid_two ()
{
    Triangulation<2> triangulation;
    
    GridGenerator::hyper_L(triangulation, 0,2);
    triangulation.refine_global (3);
    
    std::ofstream out ("L_two.vtk");
    GridOut grid_out;
    grid_out.write_vtk (triangulation, out);
    std::cout << "L shape mesh in 2d is written to L.vtk" << std::endl;
}


void shell_three ()
{

  Triangulation<3> triangulation;

  const Point<3> center (1,0,0);
  const double inner_radius = 0.5,
               outer_radius = 1.0;
  GridGenerator::hyper_shell (triangulation,
                              center, inner_radius, outer_radius,
                              96,false);
  const HyperShellBoundary<3> boundary_description(center);
  triangulation.set_boundary (0, boundary_description);

  for (unsigned int step=0; step<2; ++step)
    {
      Triangulation<3>::active_cell_iterator cell = triangulation.begin_active(), endc = triangulation.end();

        for (; cell!=endc; ++cell)
        for (unsigned int v=0;
             v < GeometryInfo<3>::vertices_per_cell;
             ++v)
          {
            const double distance_from_center
              = center.distance (cell->vertex(v));

            if (std::fabs(distance_from_center - inner_radius) > 1e-10 ||
            		std::fabs(distance_from_center - outer_radius) < 1e-10 ||
                std::fabs(distance_from_center - (inner_radius+(inner_radius-outer_radius)/2))<1e-10)
              {
                cell->set_refine_flag ();
                break;
              }
          }

      triangulation.execute_coarsening_and_refinement ();
    }


  std::ofstream out ("shell-3d.vtk");
  GridOut grid_out;
  grid_out.write_vtk (triangulation, out);

  std::cout << "Shell mesh in 3d is written to shell-3d.vtk" << std::endl;

  triangulation.set_boundary (0);
}

void ball_grid ()
{
    Triangulation<3> triangulation;
    const Point<3> center(0,0,0);
    const double radius = 0.5;
    GridGenerator:: hyper_ball(triangulation,center,radius);
    const SphericalManifold<3> manifold_description(center);
    triangulation.set_manifold(0,manifold_description);

    for (unsigned int step=0; step<5; ++step)
    {
        Triangulation<3>:: active_cell_iterator
        cell = triangulation.begin_active(),endc = triangulation.end();
        for (; cell!=endc; ++cell)
            for(unsigned int v=0; v < GeometryInfo<3>::vertices_per_cell; ++v)
            {
                const double distance_from_center = center.distance(cell -> vertex(v));
                if (std::fabs(distance_from_center - radius) < 1e-10 ||
                    std::fabs(distance_from_center - (radius)/2)<1e-10)
                {
                    cell->set_refine_flag();
                    break;
                }
                
            }
        triangulation.execute_coarsening_and_refinement();
    }


    std::ofstream out ("ball_3d.vtk");
    GridOut grid_out;
    grid_out.write_vtk (triangulation, out);
    
    std::cout << "Ball mesh in 3d is written to ball_3d.vtk" << std::endl;
    
    triangulation.set_manifold (0);
    

}

void shell_two_one()
{
    Triangulation<2> mesh;
    const Point<2> center(1,0);
    const double inner_radius = 0.5, outer_radius = 2.0;
    GridGenerator::hyper_shell(mesh, center,inner_radius,outer_radius,10);
    const HyperShellBoundary<2> boundary_description(center);
    mesh.set_boundary(0,boundary_description);
    for (unsigned int step=0; step<8; ++step)
    {
        Triangulation<2>:: active_cell_iterator
        cell = mesh.begin_active(),endc = mesh.end();
        for (; cell!=endc; ++cell)
            for(unsigned int v=0; v < GeometryInfo<2>::vertices_per_cell; ++v)
            {
                const double distance_from_center = center.distance(cell -> vertex(v));
                if (std::fabs(distance_from_center - inner_radius) < 1e-10 ||
                    std::fabs(distance_from_center - outer_radius) < 1e-10 ||
                    std::fabs(distance_from_center - (inner_radius+(outer_radius-inner_radius)/2))<1e-10 ||
                    std::fabs(distance_from_center - (inner_radius+(outer_radius-inner_radius)/4))<1e-10 ||
                    std::fabs(distance_from_center - (inner_radius+(outer_radius-inner_radius)/6))<1e-10)
                {
                    cell->set_refine_flag();
                   break;
                }
               
            }
        mesh.execute_coarsening_and_refinement();
    }
    std::ofstream out ("shell_2d_1.eps");
    GridOut grid_out;
    grid_out.write_eps (mesh, out);
    
    std::cout << "Shell mesh in 2d_1 is written to shell_2d_1.eps" << std::endl;
    
    mesh.set_boundary (0);
}

void shell_two_two()
{
    Triangulation<2> mesh;
    const Point<2> center(1,0);
    const double inner_radius = 0.5, outer_radius = 2.0;
    GridGenerator::hyper_shell(mesh, center,inner_radius,outer_radius,10);
    const HyperShellBoundary<2> boundary_description(center);
    mesh.set_boundary(0,boundary_description);
    for (unsigned int step=0; step<8; ++step)
    {
        Triangulation<2>:: active_cell_iterator
        cell = mesh.begin_active(),endc = mesh.end();
        for (; cell!=endc; ++cell)
        {
            for(unsigned int v=0; v < GeometryInfo<2>::vertices_per_cell; ++v)
            {
                const double distance_from_center = center.distance(cell -> vertex(v));
                if (std::fabs(distance_from_center - inner_radius) < 1e-10 ||
                    std::fabs(distance_from_center - outer_radius) < 1e-10 ||
                    std::fabs(distance_from_center - (inner_radius+(outer_radius-inner_radius)/2))<1e-10 ||
                    std::fabs(distance_from_center - (inner_radius+(outer_radius-inner_radius)/4))<1e-10 ||
                    std::fabs(distance_from_center - (inner_radius+(outer_radius-inner_radius)/6))<1e-10)
                {
                    cell->set_refine_flag();
                    break;
                }
                
            }
        }
        mesh.execute_coarsening_and_refinement();
    }
    std::ofstream out ("shell_2d_2.eps");
    GridOut grid_out;
    grid_out.write_eps (mesh, out);
    
    std::cout << "Shell mesh in 2d_2 is written to shell_2d_2.eps" << std::endl;
    
    mesh.set_boundary (0);
}


void second_grid ()
{
    Triangulation<2> triangulation;
    const Point<2> center (1,0);
    const double inner_radius = 0.5,
    outer_radius = 1.0;
    GridGenerator::hyper_shell (triangulation,
                                center, inner_radius, outer_radius,
                                10);
    triangulation.set_all_manifold_ids(0);
    const SphericalManifold<2> manifold_description(center);
    triangulation.set_manifold (0, manifold_description);
    for (unsigned int step=0; step<5; ++step)
    {
        Triangulation<2>::active_cell_iterator
        cell = triangulation.begin_active(),
        endc = triangulation.end();
        for (; cell!=endc; ++cell)
        {
            for (unsigned int v=0;
                 v < GeometryInfo<2>::vertices_per_cell;
                 ++v)
            {
                const double distance_from_center
                = center.distance (cell->vertex(v));
                if (std::fabs(distance_from_center - inner_radius) < 1e-10 ||
                    std::fabs(distance_from_center - outer_radius) < 1e-10 ||
                    std::fabs(distance_from_center-(inner_radius+(outer_radius-inner_radius)/3))<1e-10/* ||
                    std::fabs(distance_from_center-(inner_radius+(outer_radius-inner_radius)/0.5))<1e-10*/)
                {
                    cell->set_refine_flag ();
                    break;
                }
            }
        }
        triangulation.execute_coarsening_and_refinement ();
    }
    std::ofstream out ("grid-2.eps");
    GridOut grid_out;
    grid_out.write_eps (triangulation, out);
    std::cout << "Grid written to grid-2.eps" << std::endl;
    triangulation.set_manifold (0);
}

/*void second_grid ()
{
    Triangulation<2> triangulation;
    const Point<2> center (1,0);
    const double inner_radius = 0.5,
    outer_radius = 1.0;
    GridGenerator::hyper_shell (triangulation,
                                center, inner_radius, outer_radius,
                                10);
    
    const HyperShellBoundary<2> boundary_description(center);
    triangulation.set_boundary(0,boundary_description);
    for (unsigned int step=0; step<5; ++step)
    {
        Triangulation<2>::active_cell_iterator
        cell = triangulation.begin_active(),
        endc = triangulation.end();
        for (; cell!=endc; ++cell)
        {
            for (unsigned int v=0;
                 v < GeometryInfo<2>::vertices_per_cell;
                 ++v)
            {
                const double distance_from_center
                = center.distance (cell->vertex(v));
                if (std::fabs(distance_from_center - inner_radius) < 1e-10||
                    std::fabs(distance_from_center-(inner_radius+(outer_radius-inner_radius)/2))<1e-10)
                {
                    cell->set_refine_flag ();
                    break;
                }
            }
        }
        triangulation.execute_coarsening_and_refinement ();
    }
    std::ofstream out ("grid-2.eps");
    GridOut grid_out;
    grid_out.write_eps (triangulation, out);
    std::cout << "Grid written to grid-2.eps" << std::endl;
    triangulation.set_boundary (0);
}*/

int main ()
{
  cube_grid ();
   shell_three ();
    shell_two_one();
     shell_two_two();
      L_grid_three ();
     L_grid_two ();
    square_grid();
   ball_grid();
  second_grid();
}

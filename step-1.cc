/* ---------------------------------------------------------------------
 * $Id: step-1.cc 33051 2014-06-16 16:08:15Z bangerth $
 *
 * Copyright (C) 1999 - 2014 by the deal.II authors
 *
 * This file is part of the deal.II library.
 *
 * The deal.II library is free software; you can use it, redistribute
 * it, and/or modify it under the terms of the GNU Lesser General
 * Public License as published by the Free Software Foundation; either
 * version 2.1 of the License, or (at your option) any later version.
 * The full text of the license can be found in the file LICENSE at
 * the top level of the deal.II distribution.
 *
 * ---------------------------------------------------------------------

 */


#include <deal.II/grid/tria.h>
#include <deal.II/grid/tria_accessor.h>
#include <deal.II/grid/tria_iterator.h>
#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/tria_boundary_lib.h>
#include <deal.II/grid/grid_out.h>

#include <iostream>

#include <fstream>
#include <cmath>
//This is the most up to dated grid code.
using namespace dealii;


void cube_grid ()
{
  Triangulation<3> triangulation;

  GridGenerator::hyper_cube (triangulation);
  triangulation.refine_global (4);

  std::ofstream out ("cube.vtk");
  GridOut grid_out;
  grid_out.write_vtk (triangulation, out);
  std::cout << "Grid written to cube.vtk" << std::endl;
}

void square_grid ()
{
    Triangulation<2> mesh;
    GridGenerator:: hyper_cube (mesh);
    mesh.refine_global(4);
    std:: ofstream out("square.eps");
    GridOut grid_out;
    grid_out.write_eps (mesh, out);
    std:: cout<<" Square Mesh is Written to square.eps"<<std::endl;
}

void L_grid ()
{
  Triangulation<3> triangulation;

  GridGenerator::hyper_L(triangulation, 0,2);
  triangulation.refine_global (3);

  std::ofstream out ("L.vtk");
  GridOut grid_out;
  grid_out.write_vtk (triangulation, out);
  std::cout << "Grid written to L.vtk" << std::endl;
}



void shell_grid ()
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
      Triangulation<3>::active_cell_iterator
      cell = triangulation.begin_active(),
      endc = triangulation.end();

      for (; cell!=endc; ++cell)
        for (unsigned int v=0;
             v < GeometryInfo<3>::vertices_per_cell;
             ++v)
          {
            const double distance_from_center
              = center.distance (cell->vertex(v));

            if (std::fabs(distance_from_center - inner_radius) > 1e-10 /*||
            		std::fabs(distance_from_center - outer_radius) < 1e-10*/)
              {
                cell->set_refine_flag ();
                break;
              }
          }

      triangulation.execute_coarsening_and_refinement ();
    }


  std::ofstream out ("shell.vtk");
  GridOut grid_out;
  grid_out.write_vtk (triangulation, out);

  std::cout << "Grid written to shell.vtk" << std::endl;

  triangulation.set_boundary (0);
}




int main ()
{
  cube_grid ();
  shell_grid ();
  L_grid ();
  square_grid();
}

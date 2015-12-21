//This code will do the numbering on a mesh.



#include <deal.II/grid/tria.h>
#include <deal.II/grid/tria_accessor.h>
#include <deal.II/grid/tria_iterator.h>
#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/manifold_lib.h>
#include <deal.II/grid/grid_out.h>
#include <deal.II/dofs/dof_handler.h>
#include <deal.II/fe/fe_q.h>
#include <deal.II/dofs/dof_tools.h>
#include <deal.II/lac/sparse_matrix.h>
#include <deal.II/lac/dynamic_sparsity_pattern.h>
#include <deal.II/dofs/dof_renumbering.h>
#include <fstream>
using namespace dealii;



const int dim =3;

void make_grid (Triangulation<dim> &triangulation)
{
    GridGenerator::hyper_cube (triangulation);
    triangulation.refine_global (5);
     if (dim==2)
     {
         std::ofstream out ("square.eps");
         GridOut grid_out;
         grid_out.write_eps(triangulation, out);
         std::cout << "Mesh in "<<dim<<"d is written to square.eps"<<std::endl;
    }
    else
    {
        std::ofstream out ("cube.vtk");
        GridOut grid_out;
        grid_out.write_vtk(triangulation, out);
        std::cout << "Mesh in "<<dim<<"d is written to cube.vtk"<<std::endl;
    }
   
}


/*void make_grid (Triangulation<dim> &triangulation)
{
    const Point<dim> center(0,0,0);
    const double radius = 0.5;
    GridGenerator:: hyper_ball(triangulation,center,radius);
    const SphericalManifold<dim> manifold_description(center);
    triangulation.set_manifold(0,manifold_description);
    
    for (unsigned int step=0; step<3; ++step)
    {
        Triangulation<dim>:: active_cell_iterator
        cell = triangulation.begin_active(),endc = triangulation.end();
        for (; cell!=endc; ++cell)
            for(unsigned int v=0; v < GeometryInfo<dim>::vertices_per_cell; ++v)
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
    std::ofstream out ("ball.vtk");
    GridOut grid_out;
    grid_out.write_vtk (triangulation, out);
    
    std::cout << "Ball mesh in "<<dim<<"d is written to ball.vtk" << std::endl;
    
    triangulation.set_manifold (0);
}*/

void distribute_dofs (DoFHandler<dim> &dof_handler)
{
    static const FE_Q<dim> finite_element(1);
    dof_handler.distribute_dofs (finite_element);
    DynamicSparsityPattern dynamic_sparsity_pattern(dof_handler.n_dofs(),
                                                    dof_handler.n_dofs());
    DoFTools::make_sparsity_pattern (dof_handler, dynamic_sparsity_pattern);
    SparsityPattern sparsity_pattern;
    sparsity_pattern.copy_from (dynamic_sparsity_pattern);
    std::ofstream out ("sparsity_pattern.1");
    sparsity_pattern.print_gnuplot (out);
}

void renumber_dofs (DoFHandler<dim> &dof_handler)
{
        DoFRenumbering::Cuthill_McKee (dof_handler);
        DynamicSparsityPattern dynamic_sparsity_pattern(dof_handler.n_dofs(),
                                                        dof_handler.n_dofs());
        DoFTools::make_sparsity_pattern (dof_handler, dynamic_sparsity_pattern);
        SparsityPattern sparsity_pattern;
        sparsity_pattern.copy_from (dynamic_sparsity_pattern);
        std::ofstream out ("sparsity_pattern.2");
        sparsity_pattern.print_gnuplot (out);
}

int main ()
{
    Triangulation<dim> triangulation;
    make_grid (triangulation);
    DoFHandler<dim> dof_handler (triangulation);
    distribute_dofs (dof_handler);
    renumber_dofs (dof_handler);
}










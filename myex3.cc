//This script is my example of s simple laplace solver in dealii.



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
#include <deal.II/base/function.h>
#include <deal.II/numerics/vector_tools.h>
#include <deal.II/numerics/matrix_tools.h>
#include <deal.II/fe/fe_values.h>
#include <deal.II/base/quadrature_lib.h>
#include <deal.II/lac/vector.h>
#include <deal.II/lac/full_matrix.h>
#include <deal.II/lac/solver_cg.h>
#include <deal.II/lac/precondition.h>
#include <deal.II/numerics/data_out.h>
#include <fstream>
#include <iostream>
using namespace dealii;

class LaplaceSolver
{
	public:
		LaplaceSolver ();
		void run ();

	private:
		void make_grid ();
		void setup_system ();
		void assemble_system ();
		void solve ();
		void output_results () const;

	Triangulation<2>	triangulation;
	FE_Q<2>			fe;
	DoFHandler<2>		dof_handler;

	SparsityPattern		sparsity_pattern;
	SparseMatrix<double>	system_matrix;

	Vector<double>		solution;
	Vector<double>		system_rhs;
};

LaplaceSolver::LaplaceSolver ()
	:
	fe(1), dof_handler (triangulation)
	{}

void LaplaceSolver::make_grid ()
	{
		GridGenerator::hyper_cube (triangulation,-1,1);
		triangulation.refine_global (5);
		
		std::cout<<"Number of active cells in the domain: "
			 <<triangulation.n_active_cells()
			 <<std::endl;
	}

void LaplaceSolver::setup_system ()
	{
		dof_handler.distribute_dofs (fe);
		std::cout << "Number of degrees of freedom: "
			  << dof_handler.n_dofs()
			  <<std::endl;

		DynamicSparsityPattern dsp(dof_handler.n_dofs());
		DoFTools::make_sparsity_pattern (dof_handler, dsp);
		sparsity_pattern.copy_from(dsp);

		system_matrix.reinit (sparsity_pattern);

		solution.reinit(dof_handler.n_dofs());
		system_rhs.reinit (dof_handler.n_dofs());
	}

void LaplaceSolver:: assemble_system ()
	{
		QGauss<2> quadrature_formula(2);
		FEValues<2> fe_values (fe, quadrature_formula, update_values | 
					update_gradients | update_JxW_values);

		const unsigned int dofs_per_cell = fe.dofs_per_cell;
		const unsigned int n_q_points 	 = quadrature_formula.size();

		FullMatrix<double>	cell_matrix (dofs_per_cell, dofs_per_cell);
		Vector<double>		cell_rhs (dofs_per_cell);

		std::vector<types::global_dof_index> local_dof_indices (dofs_per_cell);

		DoFHandler<2>:: active_cell_iterator
		cell = dof_handler.begin_active(), endc = dof_handler.end();

		for (; cell! = endc; ++cell)
		{	
			fe_values.reinit (cell);
			cell_matrix = 0;
			cell_rhs = 0;
			
			for (unsigned int q_index=0; q_index<n_q_points; ++q_index)
			{
				for (unsigned int i=0; i<dofs_per_cell; ++i)
				for (unsigned int j=0; j<dofs_per_cell; ++j)
				cell_matrix(i,j) += (fe_values.shape_grad (i, q_index) *
						     fe_values.shape_grad (j, q_index) *
						     fe_values.JxW (q_index));

				for (unsigned int i=0; i<dofs_per_cell; ++i)
				cell_rhs(i) += (fe_values.shape_value (i, q_index) *
						1 * fe_values.JxW (q_index));
			}
			
			cell->get_dof_indices (local_dof_indices);

				for (unsigned int i=0; i<dofs_per_cell; ++i)
				for (unsigned int j=0; j<dofs_per_cell; ++j)
				system_matrix.add (local_dof_indices[i],
					           local_dof_indices[j],
				          	    cell_matrix(i,j));

				for (unsigned int i=0; i<dofs_per_cell; ++i)
				system_rhs(local_dof_indices[i]) += cell_rhs(i);
		}
		
		std::map<types::global_dof_index,double> boundary_values;
		VectorTools::interpolate_boundary_values (dof_handler,0,ZeroFunction<2>,
							  boundary_values);

		MatrixTools::apply_boundary_values (boundary_values, system_matrix,
						    solution, system_rhs);
	}

void LaplaceSolver:: solve()
	{
		SolverControl	solver_control (1000, 1e-10);
		SolverCG<>	solver (solver_control);

		solver.solve (system_matrix, solution, system_rhs,PreconditionIdentity());

	}

void LaplaceSolver:: output_results () const
	{
		DataOut<2> data_out;
		data_out.attach_dof_handler (dof_handler);
		data_out.add_data_vector (solution, "solution");

		data_out.build_patches ();
		std::ofstream output("solution.gpl");
		data_out.write_gnuplot (output);
	}

void LaplaceSolver::run();
	{
		make_grid ();
		setup_system ();
		assemble_system ();
		solve ();
		output_results ();
	}

int main ()
	{
		LaplaceSolver execute;
		execute.run();
		
		return 0;
	}

		


			



		










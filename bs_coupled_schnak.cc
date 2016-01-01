/*This script solves bulk surface reaction diffusion system*/
#include <deal.II/grid/tria.h>
#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_in.h>
#include <deal.II/grid/grid_out.h>
#include <deal.II/grid/grid_tools.h>
#include <deal.II/grid/tria_boundary_lib.h>
#include <deal.II/grid/tria_accessor.h>
#include <deal.II/grid/tria_iterator.h>
#include <deal.II/fe/mapping_q_eulerian.h>
#include <deal.II/fe/fe_system.h>
#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/fe_values.h>
#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_tools.h>
#include <deal.II/dofs/dof_renumbering.h>
#include <deal.II/dofs/dof_accessor.h>
#include <deal.II/base/quadrature_lib.h>
#include <deal.II/base/function.h>
#include <deal.II/base/data_out_base.h>
#include <deal.II/base/parameter_handler.h>
#include <deal.II/base/logstream.h>
#include <deal.II/base/thread_management.h>
#include <deal.II/base/multithread_info.h>
#include <deal.II/numerics/vector_tools.h>
#include <deal.II/numerics/vector_tools.templates.h>
#include <deal.II/numerics/matrix_tools.h>
#include <deal.II/numerics/fe_field_function.h>
#include <deal.II/numerics/data_out.h>
#include <deal.II/lac/vector.h>
#include <deal.II/lac/full_matrix.h>
#include <deal.II/lac/sparse_matrix.h>
#include <deal.II/lac/chunk_sparsity_pattern.h>
#include <deal.II/lac/petsc_parallel_sparse_matrix.h>
#include <deal.II/lac/block_vector.h>
#include <deal.II/lac/block_sparse_matrix.h>
#include <deal.II/lac/block_vector.h>
#include <deal.II/lac/precondition.h>
#include <deal.II/lac/solver_cg.h>
#include <deal.II/lac/solver_gmres.h>
#include <deal.II/lac/solver_bicgstab.h>
#include <deal.II/lac/sparse_ilu.h>
#include <deal.II/lac/precondition.h>
#include <cstdlib>
#include <ctime> 
#include <list>
#include <fstream>
#include <iostream>
#include <sstream>
#include <sys/stat.h>
#include <sys/types.h>

using namespace dealii;


void here(const int &number)
{ std::cout << "Here at #" << number << std::endl;}

void here(const std::string &string)
{ std::cout << "Here at " << string << std::endl;}

void here()
{  std::cout << "Here!" << std::endl;}

void dotted_line()
{
 std::cout << std::endl 
           << "--------------------------------------------------------"
           << std::endl;
}

template<typename T>
void print_is(std::string s, T value)
{
 std::cout << s << " is " 
           << value << std::endl;
}


// use this structure to make 'true' and 'false' types
// to allow function overloading in RD class 
template<bool value>
struct ValueToType
{
  enum { Value = value };
};


// have BE, FST and CN as enum
enum TimeStepMethod{ BE,FST,CN };
enum NonlinearSolver{ NEWTON, PICARD };





  //////////////////////
 //  Ellipse map  // 
//////////////////////

// map unit sphere to ellipse with 
// (x/a)^2 + (y/b)^2 + (z/c)^2 = 1

double gl_ellipse_a, gl_ellipse_b, gl_ellipse_c;

template <int dim>
Point<dim> ellipse_map (const Point<dim> &p)
{
  Point<dim> q=p;
 
  q[0] /= gl_ellipse_a;
  q[1] /= gl_ellipse_b;

   if (dim == 3) 
      q[2] /= gl_ellipse_c;

  return q;
}


  ///////////////////////////////////////
 //  Class to hold common parameters  //
///////////////////////////////////////
class CommonParameters
{
   public:
    CommonParameters(std::string &filename):filename(filename){};

    std::string filename;
    void declare_parameters();
    void get_parameters(); 
    void print_parameters();

    ParameterHandler parameter_handler;
    unsigned int     spacedim;
    std::string      testID; // timestep_method;
    TimeStepMethod   timestep_method;
    unsigned int     n_threads; 
    bool             save_animation;
    bool             read_mesh;
    double           tau, T, t, theta;
    std::string      domain_shape;
    double           ellipse_a, ellipse_b, ellipse_c;
    double           alpha_1, alpha_2,
                     beta_1, beta_2,
                     kappa_1, kappa_2;
    unsigned int     max_nonlin_iter;  // number of points in gauss quadrature rule 
    NonlinearSolver  nonlin_solver;
    double           nonlin_solver_tol,matrix_solver_tol; 

};


  ///////////////////////// 
 // Declare parameters  //
///////////////////////// 

// #declare_parameters# //
void CommonParameters::declare_parameters()
{
 parameter_handler.declare_entry ("Test ID", "",
                                   Patterns::Anything(),
                                   "The name of the test");

 parameter_handler.declare_entry ("Number of threads", "1",
                                    Patterns::Integer(),
                                    "Number of threads for parallel implementation");

 parameter_handler.declare_entry ("Space Dimension", "3",
                                    Patterns::Integer(),
                                    " ");

 parameter_handler.declare_entry ("Save Animation", "false",
                                    Patterns::Bool(),
                                   "Save .vtu solutions periodically for an animation "
                                   "(can use a lot of memory)");

 parameter_handler.declare_entry ("Read Mesh", "false",
                                    Patterns::Bool(),
                                   " ");

// declare time parameters
parameter_handler.enter_subsection("Time");	
 {
 parameter_handler.declare_entry ("Timestep Method", "FST",
                                    Patterns::Selection("BE|FST|CN"),
                                    "Time discretisation method - BE or FST or CN");
								   
 parameter_handler.declare_entry ("Timestep", ".001",
                                    Patterns::Double(),
                                    "Size of the timestep");

 parameter_handler.declare_entry ("Endtime", "10000.",
                                    Patterns::Double(),
                                    "Endtime value");
								   
 parameter_handler.declare_entry ("Theta", "0.",
                                    Patterns::Double(),
                                    "Value of theta for FST");
 }
parameter_handler.leave_subsection();

// declare mesh parameters
parameter_handler.enter_subsection("Mesh and Domain");
 {
 parameter_handler.declare_entry ("Domain Shape", "square",
                                   Patterns::Selection("square|circle|ellipse|moebius"),
                                   "Solve on a square/cube");

   parameter_handler.enter_subsection("Ellipse");	
     {
       parameter_handler.declare_entry ("Ellipse a", "1.",
                                       Patterns::Double(),
                                       "Ellipse constant a");
       parameter_handler.declare_entry ("Ellipse b", "1.",
                                       Patterns::Double(),
                                       "Ellipse constant b");
       parameter_handler.declare_entry ("Ellipse c", "1.",
                                       Patterns::Double(),
                                       "Ellipse constant c");
     }
    parameter_handler.leave_subsection(); 
 }
parameter_handler.leave_subsection();

// declare reaction parameters
parameter_handler.enter_subsection("Reaction");
 {
     parameter_handler.declare_entry ("Alpha 1", "1.",
                                       Patterns::Double(),
                                       "");
     parameter_handler.declare_entry ("Alpha 2", "1.",
                                       Patterns::Double(),
                                       "");
     parameter_handler.declare_entry ("Beta 1", "1.",
                                       Patterns::Double(),
                                       "");
     parameter_handler.declare_entry ("Beta 2", "1.",
                                       Patterns::Double(),
                                       "");
     parameter_handler.declare_entry ("Kappa 1", "1.",
                                       Patterns::Double(),
                                       "");
     parameter_handler.declare_entry ("Kappa 2", "1.",
                                       Patterns::Double(),
                                       "");
 }
parameter_handler.leave_subsection();

// declare solver parameters 
parameter_handler.enter_subsection("Solver");
 {
 parameter_handler.declare_entry ("Maximum nonlinear iterations", "50",
                                    Patterns::Integer(),
                                    "Maximum number of iterations in nonlinear solver");

 parameter_handler.declare_entry ("Nonlinear Solver", "Newton",
                                   Patterns::Selection("Newton|Picard"),
                                    " ");

 parameter_handler.declare_entry ("Nonlinear solver tolerance", "1.e-10",
                                    Patterns::Double(),
                                    "Convergence criterion for nonlinear solver");

 parameter_handler.declare_entry ("Matrix solver tolerance", "1.e-10",
                                    Patterns::Double(),
                                    " ");

  }
parameter_handler.leave_subsection();

}


  //////////////////////////////
 //  Get parameters from file //
///////////////////////////////

// Get parameters from file specified on command line
void CommonParameters::get_parameters()
{
 // give prm the filename
 parameter_handler.read_input (filename);


testID    = parameter_handler.get ("Test ID");
n_threads = parameter_handler.get_integer ("Number of threads");
save_animation = parameter_handler.get_bool ("Save Animation");
spacedim = parameter_handler.get_integer ("Space Dimension");

read_mesh = parameter_handler.get_bool ("Read Mesh");

// get time parameters
parameter_handler.enter_subsection("Time");
	std::string tmp_string  = parameter_handler.get("Timestep Method");
          if(tmp_string == "BE")
             timestep_method = BE;
          else if (tmp_string == "FST")
             timestep_method = FST;
          else if (tmp_string == "CN")
             timestep_method = CN;

	tau       = parameter_handler.get_double ("Timestep");
	T         = parameter_handler.get_double ("Endtime");
	theta     = parameter_handler.get_double ("Theta");
parameter_handler.leave_subsection();

// get mesh parameters  
parameter_handler.enter_subsection("Mesh and Domain");
	domain_shape = parameter_handler.get ("Domain Shape");
          parameter_handler.enter_subsection("Ellipse");
             ellipse_a = parameter_handler.get_double ("Ellipse a");
             ellipse_b = parameter_handler.get_double ("Ellipse b");
             ellipse_c = parameter_handler.get_double ("Ellipse c");
          parameter_handler.leave_subsection();
parameter_handler.leave_subsection();

parameter_handler.enter_subsection("Reaction");
alpha_1 = parameter_handler.get_double ("Alpha 1"); 
alpha_2 = parameter_handler.get_double ("Alpha 2"); 
beta_1  = parameter_handler.get_double ("Beta 1"); 
beta_2  = parameter_handler.get_double ("Beta 2"); 
kappa_1 = parameter_handler.get_double ("Kappa 1"); 
kappa_2 = parameter_handler.get_double ("Kappa 2"); 
parameter_handler.leave_subsection();

// get solver parameters
parameter_handler.enter_subsection("Solver");
	max_nonlin_iter   = parameter_handler.get_integer ("Maximum nonlinear iterations");
        {
         std::string tmp_string = parameter_handler.get("Nonlinear Solver");
           if(tmp_string == "Newton")
              nonlin_solver = NEWTON;
           else if (tmp_string == "Picard")
              nonlin_solver = PICARD;
        }
	nonlin_solver_tol = parameter_handler.get_double ("Nonlinear solver tolerance");
	matrix_solver_tol = parameter_handler.get_double ("Nonlinear solver tolerance");
parameter_handler.leave_subsection();

   if ( theta == -1. )
    {
     std::printf("Using default value of theta = 1-sqrt(1/2)\n");
     theta = 1.-numbers::SQRT1_2;
    }      

}

  ///////////////////////
 // Print parameters  //
////////////////////////

// #print_parameters# //
void CommonParameters::print_parameters() 
{
std::stringstream message;
message << "==============================="    << std::endl  
        << "Running with common parameters: "   << std::endl
        << "==============================="    << std::endl  
        << " Test ID         : " << testID         << std::endl 
        << " n threads       : " << n_threads      << std::endl
        << " Save Animation  : " << save_animation << std::endl
        << " Domain Shape    : " << domain_shape   << std::endl;
 if (domain_shape == "ellipse")  {
message << " Ellipse a       : " << gl_ellipse_a         << std::endl
        << " Ellipse b       : " << gl_ellipse_b         << std::endl
        << " Ellipse c       : " << gl_ellipse_c         << std::endl;
    }
message	<< " Timestep method : " << timestep_method  << std::endl
        << " Timestep        : " << tau              << std::endl 
	    << " Endtime         : " << T                << std::endl
        << " Theta           : " << theta            << std::endl
        << " alpha 1         : " << alpha_1          << std::endl
        << " beta  1         : " << beta_1           << std::endl
        << " kappa 1         : " << kappa_1          << std::endl
        << " alpha 2         : " << alpha_2          << std::endl
        << " beta  2         : " << beta_2           << std::endl
        << " kappa 2         : " << kappa_2          << std::endl
        << "============================="    << std::endl ;  
			   
std::cout << message.str();

//std::string tmp_filename = output_directory + "/" + testID+".log";
//std::ofstream output_stream(tmp_filename.c_str());

//output_stream << message.str(); 


}

  /////////////////////////////////////////////
 //  Class to handler parameters from file  //
/////////////////////////////////////////////
class Parameters
{
   public:
    Parameters(std::string &filename):filename(filename){};


    std::string filename;
    void declare_parameters();
    void get_parameters(); 

    ParameterHandler parameter_handler;
    unsigned int     fe_degree,mapping_degree;
    unsigned int     meshsize;
    double           a, b, d, gamma;//, u_eq, v_eq;
    unsigned int     n_gauss_quad, max_nonlin_iter;  // number of points in gauss quadrature rule 
    NonlinearSolver  nonlin_solver;
    double           nonlin_solver_tol,matrix_solver_tol; 

};


  ///////////////////////// 
 // Declare parameters  //
///////////////////////// 

// #declare_parameters# //
void Parameters::declare_parameters()
{

// declare mesh parameters
parameter_handler.enter_subsection("Mesh and Domain");
 {
 parameter_handler.declare_entry ("Mesh size", "1",
                                    Patterns::Integer(),
                                    "Number of global mesh refinements");
 }
parameter_handler.leave_subsection();

						   
// declare reaction parameters
parameter_handler.enter_subsection("Reaction");
 {
 parameter_handler.declare_entry ("a", "1.", 
                                    Patterns::Double(),
                                    "Reaction parameter, a");
 
 parameter_handler.declare_entry ("b", "1.",
                                    Patterns::Double(),
                                    "Reaction parameter, b");

 parameter_handler.declare_entry ("d", "1.",
                                    Patterns::Double(),
                                    "Reaction parameter, d");

 parameter_handler.declare_entry ("gamma", "1.",
                                    Patterns::Double(),
                                    "Reaction parameter, gamma");

 /*parameter_handler.declare_entry ("u equilibrium", "",
                                    Patterns::Double(),
                                    "Equilibrium value of u");

 parameter_handler.declare_entry ("v equilibrium", "",
                                    Patterns::Double(),
                                    "Equilibrium value of v");*/
 }
parameter_handler.leave_subsection();

// declare solver parameters 
parameter_handler.enter_subsection("Solver");
 {
 parameter_handler.declare_entry ("Number of gauss quadrature points", "2",
                                    Patterns::Integer(),
                                    "Number of points for gauss quadrature");

 parameter_handler.declare_entry ("Maximum nonlinear iterations", "50",
                                    Patterns::Integer(),
                                    "Maximum number of iterations in nonlinear solver");

 parameter_handler.declare_entry ("Nonlinear Solver", "Newton",
                                   Patterns::Selection("Newton|Picard"),
                                    " ");

 parameter_handler.declare_entry ("Nonlinear solver tolerance", "1.e-10",
                                    Patterns::Double(),
                                    "Convergence criterion for nonlinear solver");

 parameter_handler.declare_entry ("Matrix solver tolerance", "1.e-10",
                                    Patterns::Double(),
                                    " ");

  }
parameter_handler.leave_subsection();

// declare fe parameters
parameter_handler.enter_subsection("Finite Element Space");
 {
  parameter_handler.declare_entry ("Space Dimension", "3",
                                          Patterns::Integer(),
                                          "Space dimension of the problem");
  parameter_handler.declare_entry ("Domain Dimension", "3",
                                          Patterns::Integer(),
                                          "Dimension of the domain");
  parameter_handler.declare_entry ("FE Polynomial Order", "1",
                                          Patterns::Integer(),
                                          "Degree of the polynomial shape functions");
  parameter_handler.declare_entry ("Mapping Order", "1",
                                          Patterns::Integer(),
                                          "Order of mapping between reference and real cells");
  }
parameter_handler.leave_subsection();
}


  //////////////////////////////
 //  Get parameters from file //
///////////////////////////////

// Get parameters from file specified on command line
void Parameters::get_parameters()
{
 // give prm the filename
 parameter_handler.read_input (filename);

// get mesh parameters  
parameter_handler.enter_subsection("Mesh and Domain");
	meshsize = parameter_handler.get_integer ("Mesh size");	
parameter_handler.leave_subsection();

// get reaction parameters
parameter_handler.enter_subsection("Reaction");
	a     = parameter_handler.get_double ("a");
	b     = parameter_handler.get_double ("b");
	d     = parameter_handler.get_double ("d");
	gamma = parameter_handler.get_double ("gamma");
	/*u_eq  = parameter_handler.get_double ("u equilibrium");
	v_eq  = parameter_handler.get_double ("v equilibrium");*/
parameter_handler.leave_subsection();

// get solver parameters
parameter_handler.enter_subsection("Solver");
	n_gauss_quad      = parameter_handler.get_integer ("Number of gauss quadrature points");
	max_nonlin_iter   = parameter_handler.get_integer ("Maximum nonlinear iterations");
 
         std::string tmp_string = parameter_handler.get("Nonlinear Solver");
           if(tmp_string == "Newton")
              nonlin_solver = NEWTON;
           else if (tmp_string == "Picard")
              nonlin_solver = PICARD;

	nonlin_solver_tol = parameter_handler.get_double ("Nonlinear solver tolerance");
	matrix_solver_tol = parameter_handler.get_double ("Nonlinear solver tolerance");
parameter_handler.leave_subsection();

// get fe parameters
parameter_handler.enter_subsection("Finite Element Space");
      fe_degree      = parameter_handler.get_integer ("FE Polynomial Order");
      mapping_degree = parameter_handler.get_integer ("Mapping Order");
parameter_handler.leave_subsection();

}


  /////////////////////////////////
 // Main class for the problem  //
//////////////////////////////////

template <int dim, int spacedim>
class RDProblem 
{
  public:
    RDProblem (std::ostream &shared_out, Parameters &parameters, CommonParameters &common_parameters);
    inline void get_solution_value(const BlockVector<double> &in_solution, const Point<spacedim> &p, Vector<double> &return_vector);
    inline void get_solution_value(const BlockVector<double> &in_solution, const std::vector<Point<spacedim> > &p, std::vector<Vector<double> > &return_vector);
    inline void get_solution_value(const BlockVector<double> &in_solution, const Point<spacedim> &p, Vector<double> &return_vector,
                                   const typename DoFHandler<dim,spacedim>::active_cell_iterator &cell);
    inline void get_solution_value(const BlockVector<double> &in_solution, const std::vector<Point<spacedim> > &p, std::vector<Vector<double> > &return_vector,
                                   const typename DoFHandler<dim,spacedim>::active_cell_iterator &cell );
    inline void get_solution_value(const BlockVector<double> &in_solution, Vector<double> &return_vector,const typename DoFHandler<dim,spacedim>::active_cell_iterator &cell,
                                   const Quadrature<dim> &unit_cell_point);
    inline void get_solution_value(const BlockVector<double> &in_solution, std::vector<Vector<double> > &return_vector,const typename DoFHandler<dim,spacedim>::active_cell_iterator &cell,
                                   const std::vector<Quadrature<dim> > &unit_cell_points);


    //void run ();
    template<int dimm, int spacedimm> friend class CoupledSystemHandler;

    
  private:
    inline void initialise();
    inline void get_parameters();
    void print_parameters();
    void make_grid(ValueToType<true>  /*dim_eq_spacedim*/);
    void make_grid(ValueToType<false> /*dim_eq_spacedim*/);
    void read_grid(ValueToType<true>  /*dim_eq_spacedim*/);
    void read_grid(ValueToType<false> /*dim_eq_spacedim*/);
    void make_dofs();
    void assemble_mass_and_laplace_matrices();
    void assemble_mass_and_laplace_matrices_part(const typename DoFHandler<dim,spacedim>::active_cell_iterator &begin,
                                                 const typename DoFHandler<dim,spacedim>::active_cell_iterator &end); 
    inline void get_B_matrix(const BlockVector<double> &xi);
    void        get_B_matrix_part(const typename DoFHandler<dim,spacedim>::active_cell_iterator &begin,
                                  const typename DoFHandler<dim,spacedim>::active_cell_iterator &end,
                                  const BlockVector<double> &xi); 	
    inline void get_constant_rhs();	
    void get_constant_rhs_part(const typename DoFHandler<dim,spacedim>::active_cell_iterator &begin,
                               const typename DoFHandler<dim,spacedim>::active_cell_iterator &end); 	

				 
    void solve_system(BlockVector<double> &return_vector);

    inline void solve_block_system_whole(BlockVector<double> &return_vector);
    inline void solve_block_system_separate(BlockVector<double> &return_vector);

    inline void compute_L2norm_of_difference(const BlockVector<double> &solution_1,
                                             const BlockVector<double> &solution_2,
                                             std::vector<double>       &return_vector);
    inline void compute_L2norm_of_difference(const BlockVector<double> &solution_1,
                                             const BlockVector<double> &solution_2,
                                             double                    &return_value);
    inline void                output_nonlin_conv  (std::string output_name) const;
    inline void                output_solution_conv(std::string output_name) const;
    void output_solution (BlockVector<double> output_solution,
                          std::string         output_name);
    void process_solution();

    void BE_initialise();
    void BE_assemble_newton();
    void CN_timestep(const BlockVector<double> &additional_rhs_terms);
    void FST_timestep_1_assemble();
    void FST_timestep_2_initialise();
    void FST_timestep_2_assemble_newton();
    inline void FST_timestep();
    inline void update_time();
						  
    // object to synchronise threads
    Threads::Mutex     assembler_lock;  
    std::ostream      *shared_out; 

    // grid for u and p
    Triangulation<dim,spacedim>   triangulation;    
    FESystem<dim,spacedim>        fe;
    DoFHandler<dim,spacedim>      dof_handler;
    MappingQ<dim,spacedim>        mapping;
    QGauss<dim>                   quadrature_formula;
    BlockSparsityPattern          block_sparsity_pattern;
    unsigned int                  n_u,n_v;

    // matrices and system_rhs 
    BlockSparseMatrix<double> mass_matrix,  // with diagonal block entries non zero
                              contra_mass_matrix,  // with off diagonal entries non zero
                              laplace_matrix,
                              B_matrix,
                              system_matrix,
                              base_lhs;
    BlockVector<double>       constant_rhs, system_rhs, base_rhs;   
    // solutions
    BlockVector<double>       solution, old_solution,
                              lin_solution;
	
   // parameters 
    Parameters       *parameters;
    CommonParameters *common_parameters;
    double           t;
    double           u_eq, v_eq;
   
    // variables
    unsigned int          time_count,total_count,nonlin_count;
    std::vector<double>   nonlin_L2diff,solution_L2diff;
    double                nonlin_L2diff_xi;
    std::string           output_directory;
    bool                  save_animation;
    std::vector<std::pair<double,std::string> > times_and_names; //for paraview .pvd file
   
};

  ///////////////////////////////////
 // Class for initial conditions  //
///////////////////////////////////

// #initial_conditions# //
template <int spacedim>
class InitialConditions : public Function<spacedim>
 {
   public:
     InitialConditions (const double &u_eq, const double &v_eq) : 
                Function<spacedim>(2),u_eq(u_eq),v_eq(v_eq) {}
 
     double u_eq,v_eq;
     virtual void vector_value (const Point<spacedim> /*&p*/,
                                Vector<double>   &values) const;
     virtual void vector_value_list (const std::vector<Point<spacedim> > &points,
                                     std::vector<Vector<double> >   &values) const;
  };

template <int spacedim>
void InitialConditions<spacedim>::vector_value (const Point<spacedim> /*&p*/,
                                                Vector<double>   &values) const
   {
     // dummy line to use input parameter
     //Point<spacedim> q=p;
     Assert (values.size() == 2,
             ExcDimensionMismatch (values.size(), 2));
     values(0) = u_eq + 0.01*std::pow(-1.,std::rand())*std::rand()/RAND_MAX;
     values(1) = v_eq + 0.01*std::pow(-1.,std::rand())*std::rand()/RAND_MAX;
   }

template <int spacedim>
void InitialConditions<spacedim>::vector_value_list (const std::vector<Point<spacedim> > &points,
                                                     std::vector<Vector<double> >   &value_list) const
   {
     // dummy line to use input parameter
     //Point<spacedim> q=p;
     const unsigned int n_points = points.size();
       for (unsigned int p=0; p<n_points; ++p)
           InitialConditions<spacedim>::vector_value (points[p], value_list[p]);
   }


  ///////////////////////////////////////
 // Constructor for ShapeOptimProblem //
///////////////////////////////////////

// #constructor# //

// - initialise fe as vector values with two components
// - initialise fe polynomial order  and 
// - associate dof handler with correspoing triangulations
// - initialise cycle counter to 0

template <int dim, int spacedim>
RDProblem<dim,spacedim>::RDProblem (std::ostream &shared_out,Parameters &parameters, CommonParameters &common_parameters) :
         shared_out(&shared_out),
         fe(FE_Q<dim,spacedim>(parameters.fe_degree), 2), 
	     dof_handler(triangulation),
         mapping(parameters.mapping_degree),   
         quadrature_formula(parameters.n_gauss_quad),          
         parameters(&parameters),
         common_parameters(&common_parameters),
		 t(0.),
         time_count(0),
         total_count(0),
         nonlin_count(0),
         nonlin_L2diff(2),
         solution_L2diff(2)
{}


  //////////////////////////
 //  Get solution values //
//////////////////////////

  // get solution value
template <int dim, int spacedim>
void RDProblem<dim,spacedim>::get_solution_value(const BlockVector<double> &in_solution, const Point<spacedim> &p, Vector<double> &return_vector)
{
    Assert(return_vector.size() == fe.n_components(),
           ExcDimensionMismatch(return_vector.size(), fe.n_components()));

    // first find the cell in which this point is, initialize 
    // a quadrature rule with it, and then a FEValues object

    typedef typename DoFHandler<dim,spacedim>::active_cell_iterator active_cell_iterator;
    const std::pair<active_cell_iterator,Point<dim> >
                 cell_point = GridTools::find_active_cell_around_point (mapping, dof_handler, p);

    Assert(GeometryInfo<dim>::distance_to_unit_cell(cell_point.second) < 1e-7,
           ExcInternalError());

    /*{
     
      Point<spacedim> real_point, distance;
       real_point = mapping.transform_unit_to_real_cell(cell_point.first,cell_point.second);
        for(unsigned int i=0;i<spacedim;++i)
          distance[i] = real_point[i]-p[i];

      std::cout << "For point " << p << " found point " << real_point
                  << " with distance " << distance.norm() << std::endl;

    }*/

    const Quadrature<dim> quadrature (GeometryInfo<dim>::project_to_unit_cell(cell_point.second));

    FEValues<dim,spacedim> fe_values(mapping, fe, quadrature, update_values);
    fe_values.reinit(cell_point.first);

    // then use this to get at the values of the given fe_function at this point
    std::vector<Vector<double> > u_value(1, Vector<double>( fe.n_components() ) );

    fe_values.get_function_values(in_solution, u_value);

    return_vector =  u_value[0];
}

template <int dim, int spacedim>
void RDProblem<dim,spacedim>::get_solution_value(const BlockVector<double> &in_solution, const std::vector<Point<spacedim> > &p, 
                                                 std::vector<Vector<double> > &return_vector)
{
    Assert(return_vector.size() == p.size(),
           ExcDimensionMismatch(return_vector.size(), p.size()));

  for (unsigned int i=0; i< p.size(); ++i)
    get_solution_value(in_solution,p[i],return_vector[i]);

}

  // get solution value
template <int dim, int spacedim>
void RDProblem<dim,spacedim>::get_solution_value(const BlockVector<double> &in_solution, const Point<spacedim> &p, Vector<double> &return_vector,
                                                 const typename DoFHandler<dim,spacedim>::active_cell_iterator &cell)
{

    Assert(return_vector.size() == fe.n_components(),
           ExcDimensionMismatch(return_vector.size(), fe.n_components()));

    Point<dim> unit_cell_point = mapping.transform_real_to_unit_cell(cell,p);

    const Quadrature<dim> quadrature (unit_cell_point);
    FEValues<dim,spacedim> fe_values(mapping, fe, quadrature, update_values);
    fe_values.reinit(cell);

    // then use this to get at the values of the given fe_function at this point
    std::vector<Vector<double> > u_value(1, Vector<double>( fe.n_components() ) );

    fe_values.get_function_values(in_solution, u_value);

    return_vector =  u_value[0];
}

template <int dim, int spacedim>
void RDProblem<dim,spacedim>::get_solution_value(const BlockVector<double> &in_solution, const std::vector<Point<spacedim> > &p, 
                                                 std::vector<Vector<double> > &return_vector, 
                                                 const typename DoFHandler<dim,spacedim>::active_cell_iterator &cell)
{
    Assert(return_vector.size() == p.size(),
           ExcDimensionMismatch(return_vector.size(), p.size()));

  for (unsigned int i=0; i< p.size(); ++i)
    get_solution_value(in_solution,p[i],return_vector[i],cell);

}


  // get solution value
template <int dim, int spacedim>
void RDProblem<dim,spacedim>::get_solution_value(const BlockVector<double> &in_solution, Vector<double> &return_vector,
                                                 const typename DoFHandler<dim,spacedim>::active_cell_iterator &cell,
                                                 const Quadrature<dim> &quad)
{

    Assert(return_vector.size() == fe.n_components(),
           ExcDimensionMismatch(return_vector.size(), fe.n_components()));

    FEValues<dim,spacedim> fe_values(mapping, fe, quad, update_values);
    fe_values.reinit(cell);

    std::vector<Vector<double> > u_value(1, Vector<double>( fe.n_components() ) );

    fe_values.get_function_values(in_solution, u_value);

    return_vector =  u_value[0];
}

template <int dim, int spacedim>
void RDProblem<dim,spacedim>::get_solution_value(const BlockVector<double> &in_solution, std::vector<Vector<double> > &return_vector, 
                                                 const typename DoFHandler<dim,spacedim>::active_cell_iterator &cell,
                                                 const std::vector<Quadrature<dim> > &quads)
{
    Assert(return_vector.size() == quads.size(),
           ExcDimensionMismatch(return_vector.size(), quads.size()));

  for (unsigned int i=0; i< quads.size(); ++i)
    get_solution_value(in_solution,return_vector[i],cell,quads[i]);

}

  ////////////////
 // Initialise //
////////////////

template <int dim, int spacedim>
void RDProblem<dim,spacedim>::initialise()
{
 get_parameters ();
 print_parameters();
 
 if(common_parameters->read_mesh)
   read_grid(ValueToType<dim==spacedim>());
  else
    make_grid(ValueToType<dim==spacedim>());
 
 make_dofs();
 assemble_mass_and_laplace_matrices();
 get_constant_rhs();
  // set up initial conditions
 VectorTools::interpolate(mapping, dof_handler, InitialConditions<spacedim>(u_eq,v_eq), solution);
 old_solution = solution;
 
  // output grid
  {
    std::string tmp_string = output_directory + "/grid.msh";
   std::ofstream out (tmp_string.c_str());
   GridOut grid_out;
   grid_out.write_msh (triangulation, out);
  }
  
   // then output initial conditions
 output_solution(solution,output_directory+"/xi-ini.vtu");
  
 }
 
  //////////////////////////////////////////
 //  Get parameters from parameter class //
//////////////////////////////////////////

// Get parameters from file specified on command line
template <int dim, int spacedim>
void RDProblem<dim,spacedim>::get_parameters()
{

  if((common_parameters->domain_shape) == "ellipse")
   {
   // get ellipse parameters - store as global variables 
     gl_ellipse_a = common_parameters -> ellipse_a;
     gl_ellipse_b = common_parameters -> ellipse_b;
     gl_ellipse_c = common_parameters -> ellipse_c;
   }


//if(dim==spacedim)
//{
  u_eq = (parameters->a)+(parameters->b);
  double apb_2 = ((parameters->a)+(parameters->b))*((parameters->a)+(parameters->b));
  v_eq = (parameters->b)/apb_2;
/*}
else if ((common_parameters->alpha_1) && (common_parameters->alpha_2)) // if dim<spacedim and denoms not zero
{
  double apb_2   = ((parameters->a)+(parameters->b))*((parameters->a)+(parameters->b));
  double apb_3   = apb_2*((parameters->a)+(parameters->b));
  u_eq = ( (common_parameters->beta_1)*apb_3+(common_parameters->kappa_1)*(parameters->b) )
            / ( (common_parameters->alpha_1)*apb_2 ); 
  v_eq = ( (common_parameters->beta_2)*apb_3+(common_parameters->kappa_2)*(parameters->b) )
            / ( (common_parameters->alpha_2)*apb_2 );   
}
else
{
  u_eq = (parameters->a)+(parameters->b);
  double apb_2 = ((parameters->a)+(parameters->b))*((parameters->a)+(parameters->b));
  v_eq = (parameters->b)/apb_2;
}
*/
}

  ///////////////////////
 // Print parameters  //
////////////////////////

// #print_parameters# //
template <int dim, int spacedim>
void  RDProblem<dim,spacedim>::print_parameters() 
{

std::stringstream message;
message << "============================="    << std::endl  
        << "Running with parameters: "        << std::endl
        << "============================="    << std::endl  
        << " Space dim       : " << spacedim                         << std::endl
        << " Domain dim      : " << dim                              << std::endl
        << " FE shape func   : " << fe.get_name()                    << std::endl 
        << " Mapping degree  : " << mapping.get_degree()             << std::endl 
        << " Mesh size       : " << parameters->meshsize             << std::endl
        << " a               : " << parameters->a                       << std::endl
        << " b               : " << parameters->b                       << std::endl
        << " d               : " << parameters->d                       << std::endl
        << " gamma           : " << parameters->gamma                   << std::endl
        << "============================="    << std::endl ;  
			   
(*shared_out) << message.str();

//std::string tmp_filename = output_directory + "/" + testID+".log";
//std::ofstream output_stream(tmp_filename.c_str());

//output_stream << message.str(); 


}

  ///////////////////////////////////
 // Make grid  for dim = spacedim // 
///////////////////////////////////

// #make_grid //

template <int dim, int spacedim>
void RDProblem<dim,spacedim>::make_grid(ValueToType<true>  /*dim_eq_spacedim*/)
{
  // dummy line to use input parameter
  //ValueToType<true> dummy = dim_eq_spacedim;

  (*shared_out) << "Making mesh..." << std::endl;

  // create mesh
  if ((common_parameters->domain_shape) == "square")
     { 
      GridGenerator::hyper_cube (triangulation, 0., 1.);  // create [0,1]^spacedim 
      triangulation.refine_global (parameters->meshsize);
     }
   
  else if ( ( (common_parameters->domain_shape) ==  "circle") || ((common_parameters->domain_shape) == "ellipse"))  
   {    
   // define centre
   // const Point<dim> centre (0,0,0);
   bool         initialize = 1;
   const        Point<spacedim> centre (initialize); //initialize all to 0
   const double radius = 1.; 
   // tell program we have a curved boundary (needed for proper mesh refinement)
   static const HyperBallBoundary<dim,spacedim> circular_boundary_description(centre,radius);
   triangulation.set_boundary (0,  circular_boundary_description); 
   GridGenerator::hyper_ball (triangulation,centre,radius);
   triangulation.refine_global (parameters->meshsize);

    if ((common_parameters->domain_shape) == "ellipse")
      GridTools::transform(&ellipse_map<dim>, triangulation);  

   }

  else if ( (common_parameters->domain_shape) ==  "shell")  
   {    
   // define centre
   bool         initialize = 1;
   const        Point<spacedim> centre (initialize); //initialize all to 0
   const double outer_radius = 1.,inner_radius=.5; 
   // tell program we have a curved boundary (needed for proper mesh refinement)
   static const HyperShellBoundary<spacedim> shell_boundary_description(centre);    
   triangulation.set_boundary (0,  shell_boundary_description); 
   GridGenerator::hyper_shell (triangulation,centre,inner_radius,outer_radius,96,false);
   triangulation.refine_global (parameters->meshsize);

   //if ((common_parameters->domain_shape) == "ellipse")
   //   GridTools::transform(&ellipse_map<dim>, triangulation);  

   }
  else if ( (common_parameters->domain_shape) ==  "moebius" )
   {
	   GridGenerator::moebius(triangulation,1000,0,1.,.5);
	   triangulation.refine_global(parameters->meshsize);	   
   }
 
 
 
 
}

  ///////////////////////////////////
 // Make grid  for dim < spacedim // 
///////////////////////////////////

// #make_grid# //

template <int dim, int spacedim>
void RDProblem<dim,spacedim>::make_grid(ValueToType<false>  /*dim_eq_spacedim*/)
{
  // dummy line to use input parameter
  //ValueToType<false> dummy = dim_eq_spacedim;

  (*shared_out) << "Making mesh..." << std::endl;

  // create mesh
  if ((common_parameters->domain_shape) == "square")
     { 
      Triangulation<spacedim> volume_mesh;
         GridGenerator::hyper_cube (volume_mesh, 0., 1.);  // create [0,1]^spacedim
          std::set<types::boundary_id> boundary_ids;
          boundary_ids.insert (0);
         GridGenerator::extract_boundary_mesh (volume_mesh, triangulation,
                                               boundary_ids);
     triangulation.refine_global (parameters->meshsize);
     }
   
  else if ( ((common_parameters->domain_shape) == "circle") || ((common_parameters->domain_shape) == "ellipse") )
   { 
   // define centre
   bool         initialize = 1;
   const        Point<spacedim> centre (initialize); //initialize all to 0
   const double radius = 1.; 
   // tell program we have a curved boundary (needed for proper mesh refinement)
   static const HyperBallBoundary<dim,spacedim> circular_boundary_description(centre,radius);    
   triangulation.set_boundary (0,  circular_boundary_description); 
       Triangulation<spacedim> volume_mesh;
       GridGenerator::hyper_ball (volume_mesh,centre,radius);
       std::set<types::boundary_id> boundary_ids;
       boundary_ids.insert (0);

   GridGenerator::extract_boundary_mesh (volume_mesh, triangulation,
                                            boundary_ids);
   triangulation.refine_global (parameters->meshsize);
        
    if ((common_parameters->domain_shape) == "ellipse")
       GridTools::transform(&ellipse_map<spacedim>, triangulation);  
  } 
 else if ( (common_parameters->domain_shape) == "shell")
   { 
   // define centre
   bool         initialize = 1;
   const        Point<spacedim> centre (initialize); //initialize all to 0
   const double outer_radius = 1.,inner_radius=.5; 
   // tell program we have a curved boundary (needed for proper mesh refinement)
   static const HyperBallBoundary<dim,spacedim> shell_boundary_description_inner(centre,inner_radius);  
   static const HyperBallBoundary<dim,spacedim> shell_boundary_description_outer(centre,outer_radius); 
  
   triangulation.set_boundary (0,  shell_boundary_description_inner); 
   triangulation.set_boundary (1,  shell_boundary_description_outer); 
       Triangulation<spacedim> volume_mesh;
       GridGenerator::hyper_shell (volume_mesh,centre,inner_radius,outer_radius,96,true);
       std::set<types::boundary_id> boundary_ids;
       boundary_ids.insert (0);
       boundary_ids.insert (1);

   GridGenerator::extract_boundary_mesh (volume_mesh, triangulation, boundary_ids);
   triangulation.refine_global (parameters->meshsize);

       
    //if ((common_parameters->domain_shape) == "ellipse")
    //   GridTools::transform(&ellipse_map<spacedim>, triangulation);  
  } 
  else if ((common_parameters->domain_shape) == "moebius")
     { 
      Triangulation<spacedim> volume_mesh;
          GridGenerator::moebius(volume_mesh,1000,0,1.,.5);
          std::set<types::boundary_id> boundary_ids;
          boundary_ids.insert (0);
      GridGenerator::extract_boundary_mesh (volume_mesh, triangulation,
                                               boundary_ids);
     triangulation.refine_global (parameters->meshsize);
     }


}



template <int dim, int spacedim>
void RDProblem<dim,spacedim>::read_grid(ValueToType<true>  /*dim_eq_spacedim*/)
{
  // dummy line to use input parameter
  //ValueToType<true> dummy = dim_eq_spacedim;

  (*shared_out) << "Readingmesh..." << std::endl;
   GridIn<spacedim,spacedim>    grid_in;

  grid_in.attach_triangulation (triangulation);
  std::ifstream input_file("sphere_mesh.msh");
  grid_in.read_msh (input_file);

 static const HyperBallBoundary<dim> boundary;
 triangulation.set_boundary (0, boundary);

 
}

  ///////////////////////////////////
 // Make grid  for dim < spacedim // 
///////////////////////////////////

// #make_grid# //

template <int dim, int spacedim>
void RDProblem<dim,spacedim>::read_grid(ValueToType<false>  /*dim_eq_spacedim*/)
{
  (*shared_out) << "Reading mesh..." << std::endl;
 
  static const HyperBallBoundary<dim,spacedim> boundary;
  triangulation.set_boundary (0, boundary);
 
   Triangulation<spacedim>  volume_mesh;
   GridIn<spacedim>         grid_in;

  grid_in.attach_triangulation (volume_mesh);
  std::ifstream input_file("sphere_mesh.msh");
  grid_in.read_msh (input_file);

   std::set<types::boundary_id> boundary_ids;
    boundary_ids.insert (0);

   GridGenerator::extract_boundary_mesh (volume_mesh, triangulation,
                                            boundary_ids);
}


  //////////////////////////////////////
 // Make DOFS and output useful info // 
//////////////////////////////////////

// #make_grid# //
template <int dim, int spacedim>
void RDProblem<dim,spacedim>::make_dofs()
{

  // distribute dof numbers and renumber based on vector component
  dof_handler.distribute_dofs (fe);
  DoFRenumbering::component_wise (dof_handler);

  //  find number of dofs per component
  std::vector<unsigned int> dofs_per_component (2);
  DoFTools::count_dofs_per_component (dof_handler, dofs_per_component);
  // find number of dofs
  n_u = dofs_per_component[0];
  n_v = dofs_per_component[1];
 
  // Create sparsity pattern for block matrix 
  const unsigned int
  n_couplings = dof_handler.max_couplings_between_dofs();
 
  // sparsity pattern has block pattern - create correct sizes
     block_sparsity_pattern.reinit (2,2);
     block_sparsity_pattern.block(0,0).reinit (n_u, n_u, n_couplings);
     block_sparsity_pattern.block(1,0).reinit (n_v, n_u, n_couplings);
     block_sparsity_pattern.block(0,1).reinit (n_u, n_v, n_couplings);
     block_sparsity_pattern.block(1,1).reinit (n_v, n_v, n_couplings);
     block_sparsity_pattern.collect_sizes();

 DoFTools::make_sparsity_pattern (dof_handler, block_sparsity_pattern);
 block_sparsity_pattern.compress();
 
 // initialize matrices and vectors 
 system_matrix.reinit(block_sparsity_pattern);
 contra_mass_matrix.reinit  (block_sparsity_pattern);
 mass_matrix.reinit  (block_sparsity_pattern);
 laplace_matrix.reinit  (block_sparsity_pattern);
 B_matrix.reinit  (block_sparsity_pattern);
 base_lhs.reinit(block_sparsity_pattern);

 solution.reinit (2);
  solution.block(0).reinit (n_u);
  solution.block(1).reinit (n_v);
  solution.collect_sizes ();

 old_solution.reinit (solution);
 lin_solution.reinit (solution);
 constant_rhs.reinit (solution);
 system_rhs.reinit (solution);
 base_rhs.reinit(solution);
  

  // find mesh size h (max cell diameter)
  double meshsize_h = 0. ;
  typename Triangulation<dim,spacedim>::active_cell_iterator  cell = triangulation.begin_active(),
                                                             endc = triangulation.end();
  
  for ( ; cell != endc; ++cell)
    {
    if ( cell->diameter() > meshsize_h )
           meshsize_h = cell->diameter();
    }

  // Output useful information in terminal
  (*shared_out) << "Number of active cells      : "
            << triangulation.n_active_cells()
            << std::endl
            << "Total number of cells       : "
            << triangulation.n_cells()      
            << std::endl
            << "Number of degrees of freedom: "
            << dof_handler.n_dofs()
            << " (" << n_u << '+' << n_v << ')' 
            << " = " << n_u + n_v 
            << std::endl
            << "Maximum cell diameter       : "
            << meshsize_h
            << std::endl
            << "======================"    
            << std::endl;



}


  //////////////////////////////////////
 //  Assemble mass and base matrices //
//////////////////////////////////////

//  #assemble_mass_and_base_matrices# //
template <int dim, int spacedim>
void RDProblem<dim,spacedim>::assemble_mass_and_laplace_matrices()
{
  // initialise matrices
  mass_matrix = 0;
  laplace_matrix = 0;
  contra_mass_matrix = 0;
  
 Threads::ThreadGroup<> threads;
 typedef typename DoFHandler<dim,spacedim>::active_cell_iterator active_cell_iterator;
 std::vector<std::pair<active_cell_iterator,active_cell_iterator> >
                thread_ranges = Threads::split_range<active_cell_iterator> (dof_handler.begin_active (),
                                                     dof_handler.end (), (common_parameters->n_threads) );
     for (unsigned int thread=0; thread<(common_parameters->n_threads); ++thread)
       threads += Threads::new_thread (&RDProblem<dim,spacedim>::assemble_mass_and_laplace_matrices_part,
                                       *this,
                                       thread_ranges[thread].first,
                                       thread_ranges[thread].second);
  threads.join_all ();
  (*shared_out) << "Memory consuption: " << std::endl 
          << "  laplace_matrix = "
          << laplace_matrix.memory_consumption()/1048576. << " MB" << std::endl      
          << "  mass_matrix = "
          << mass_matrix.memory_consumption()/1048576. << " MB"  << std::endl
          << "======================"    << std::endl  ;  
}

   //////////////////////////////////////////////////
  //  Assemble individual parts of mass and base  //
 //       matrices in parallel implementation    //
//////////////////////////////////////////////////

//  #assemble_mass_and_base_matrices_part# //
template <int dim, int spacedim>
void RDProblem<dim,spacedim>::assemble_mass_and_laplace_matrices_part(const typename DoFHandler<dim,spacedim>::active_cell_iterator &begin,
                                                                      const typename DoFHandler<dim,spacedim>::active_cell_iterator &end)
{
  // create objects that hold information about 2d shape functions
  FEValues<dim,spacedim> fe_values (mapping, fe, quadrature_formula, 
   			                  update_values | update_gradients |
                              update_quadrature_points | update_JxW_values);
  // useful synonyms 
  const unsigned int         dofs_per_cell     = fe.dofs_per_cell;
  const unsigned int         n_q_points        = quadrature_formula.size();
  std::vector<unsigned int>  local_dof_indices  (dofs_per_cell);

  // declare local contributions to global LHS and RHS
  FullMatrix<double>   cell_matrix_laplace(dofs_per_cell, dofs_per_cell);
  FullMatrix<double>   cell_matrix_mass(dofs_per_cell, dofs_per_cell); 
   FullMatrix<double>  cell_matrix_contra_mass(dofs_per_cell, dofs_per_cell);

  // name the different components of the shape functions
  const FEValuesExtractors::Scalar u (0);
  const FEValuesExtractors::Scalar v (1);

  typename DoFHandler<dim,spacedim>::active_cell_iterator cell;
  
  // loop over all cells
  for (cell=begin; cell!=end; ++cell)
    {
      // reset values each loop
      cell_matrix_laplace = 0;
      cell_matrix_mass = 0;
      cell_matrix_contra_mass = 0;
      fe_values.reinit (cell);

      // loop over quadrature points
      for (unsigned int q_point=0; q_point<n_q_points; ++q_point)
         {		 
           // query jacobian and weight only once
          double JxW    =    fe_values.JxW(q_point);            
         for (unsigned int i=0; i<dofs_per_cell; ++i)
 	       for (unsigned int j=0; j<dofs_per_cell; ++j)
              {
             // define coefficients of matrix (query values only once)
              const double mass_u = fe_values[u].value (i, q_point)*
                                    fe_values[u].value (j, q_point);

              const double mass_v = fe_values[v].value (i, q_point)*
                                    fe_values[v].value (j, q_point);
                                                  
              const double contra_mass_u = fe_values[u].value (i, q_point)*
                                           fe_values[v].value (j, q_point);

              const double contra_mass_v = fe_values[v].value (i, q_point)*
                                           fe_values[u].value (j, q_point);
  
              const double laplace_u = fe_values[u].gradient (i, q_point)*
                                       fe_values[u].gradient (j, q_point);

              const double laplace_v = fe_values[v].gradient (i, q_point)*
                                       fe_values[v].gradient (j, q_point);


              
              cell_matrix_mass(i,j) += (mass_u+mass_v)*JxW;
              cell_matrix_contra_mass(i,j) += (contra_mass_u+contra_mass_v)*JxW;
              cell_matrix_laplace(i,j) += (laplace_u + laplace_v)*JxW; 
              }  //end for dof j
         }  // end for q_point
	  
      // add local contributions to global system
      cell->get_dof_indices (local_dof_indices);

      // get assembler lock since writing to global object
      assembler_lock.acquire ();
      for (unsigned int i=0; i<dofs_per_cell; ++i)
        for (unsigned int j=0; j<dofs_per_cell; ++j)
           {
	    laplace_matrix.add (local_dof_indices[i],
			        local_dof_indices[j],
			         cell_matrix_laplace(i,j));

     	mass_matrix.add (local_dof_indices[i],
			       local_dof_indices[j],
			       cell_matrix_mass(i,j));
			       
	    contra_mass_matrix.add (local_dof_indices[i],
			       local_dof_indices[j],
			       cell_matrix_contra_mass(i,j));
            }
     // finished writing to global object, so release assembler lock
     assembler_lock.release ();
   } //end for cell
  
} 


  /////////////////////////////
 //  Assemble newton system //
/////////////////////////////

//  #newton_assemble_system# //
template <int dim, int spacedim>
void RDProblem<dim,spacedim>::get_constant_rhs()
{
  // B matrix is BlockMatrix with
  //   - B(0,0) = B(u,u) B(0,1) = B(u,v) etc. using u and v from the input xi
 constant_rhs = 0;

 Threads::ThreadGroup<> threads;
 typedef typename DoFHandler<dim,spacedim>::active_cell_iterator active_cell_iterator;
 std::vector<std::pair<active_cell_iterator,active_cell_iterator> >
                thread_ranges = Threads::split_range<active_cell_iterator> (dof_handler.begin_active (),
                                                     dof_handler.end (), (common_parameters->n_threads));
     for (unsigned int thread=0; thread<(common_parameters->n_threads); ++thread)
       threads += Threads::new_thread (&RDProblem<dim,spacedim>::get_constant_rhs_part,
                                       *this,
                                       thread_ranges[thread].first,
                                       thread_ranges[thread].second);
  threads.join_all ();
}

  //////////////////////////////////////////////////////
 //  Get individual parts of the constant rhs term   //
//////////////////////////////////////////////////////

//  #newton_assemble_system_part# //
template <int dim, int spacedim>
void RDProblem<dim,spacedim>::get_constant_rhs_part(const typename DoFHandler<dim,spacedim>::active_cell_iterator &begin,
                                                    const typename DoFHandler<dim,spacedim>::active_cell_iterator &end)
{
  // create objects that hold information about 2d shape functions
  FEValues<dim,spacedim> fe_values (mapping, fe, quadrature_formula, 
   			            update_values | update_quadrature_points | update_JxW_values);
  // useful synonyms 
  const unsigned int                               dofs_per_cell     = fe.dofs_per_cell;
  const unsigned int                               n_q_points        = quadrature_formula.size();
  std::vector<unsigned int>                        local_dof_indices  (dofs_per_cell);

  // declare local contributions to global LHS and RHS
  Vector<double>   cell_rhs (dofs_per_cell);

  // name the different components of the shape functions
  const FEValuesExtractors::Scalar u (0);
  const FEValuesExtractors::Scalar v (1);

  typename DoFHandler<dim,spacedim>::active_cell_iterator cell;
  
  // loop over all cells
  for (cell=begin; cell!=end; ++cell)
    {
      // reset values each loop
      cell_rhs = 0;
      fe_values.reinit (cell);

      // loop over quadrature points
      for (unsigned int q_point=0; q_point<n_q_points; ++q_point)
         {	
           // query Jacobian and weight only once
           double JxW = fe_values.JxW(q_point);
		  
          for (unsigned int i=0; i<dofs_per_cell; ++i)
          {
          const double phi_i_u = fe_values[u].value (i, q_point);
          const double phi_i_v = fe_values[v].value (i, q_point);

                 // assemble 'non linear' LHS matrices
               cell_rhs(i) +=  ( (  (parameters->a)*(parameters->gamma)*phi_i_u 
                                 +  (parameters->b)*(parameters->gamma)*phi_i_v )*JxW );
        } //end for dof i
      }  // end for q_point
	  
      // add local contributions to global system
      cell->get_dof_indices (local_dof_indices);

      // get assembler lock since writing to global object
      assembler_lock.acquire ();
      for (unsigned int i=0; i<dofs_per_cell; ++i)
            constant_rhs(local_dof_indices[i]) += cell_rhs(i);
     // finished writing to global object, so release assembler lock
     assembler_lock.release ();
   } //end for cell
  
} 


  ///////////////////
 //  Get B matrix //
///////////////////

//  #get_B_matrix# //
template <int dim, int spacedim>
void RDProblem<dim,spacedim>::get_B_matrix(const BlockVector<double> &xi)
{
  // B matrix is BlockMatrix with
  //   - B(0,0) = B(u,u) B(0,1) = B(u,v) etc. using u and v from the input xi

 B_matrix = 0;
 Threads::ThreadGroup<> threads;
 typedef typename DoFHandler<dim,spacedim>::active_cell_iterator active_cell_iterator;
 std::vector<std::pair<active_cell_iterator,active_cell_iterator> >
                thread_ranges = Threads::split_range<active_cell_iterator> (dof_handler.begin_active (),
                                                     dof_handler.end (), (common_parameters->n_threads));
     for (unsigned int thread=0; thread<(common_parameters->n_threads); ++thread)
       threads += Threads::new_thread (&RDProblem<dim,spacedim>::get_B_matrix_part,
                                       *this,
                                       thread_ranges[thread].first,
                                       thread_ranges[thread].second,
                                       xi);
  threads.join_all ();
}

  ////////////////////////////////////////
 //  Get individual parts of B matrix  //
////////////////////////////////////////

//  #get_B_matrix_part# //
template <int dim, int spacedim>
void RDProblem<dim,spacedim>::get_B_matrix_part(const typename DoFHandler<dim,spacedim>::active_cell_iterator &begin,
                                                const typename DoFHandler<dim,spacedim>::active_cell_iterator &end,
                                                const BlockVector<double> &xi)
{
  // create objects that hold information about 2d shape functions
  FEValues<dim,spacedim> fe_values (mapping, fe, quadrature_formula, 
   			      update_values | update_gradients |
                              update_quadrature_points | update_JxW_values);
  // useful synonyms 
  const unsigned int                               dofs_per_cell     = fe.dofs_per_cell;
  const unsigned int                               n_q_points        = quadrature_formula.size();
  std::vector<unsigned int>                        local_dof_indices  (dofs_per_cell);

  // declare local contributions to global LHS and RHS
  FullMatrix<double>   cell_matrix (dofs_per_cell, dofs_per_cell);

  // store local values of u and v in local vectors
   std::vector<double>               u_loc(n_q_points), v_loc(n_q_points);    
   std::vector<Vector<double> >      xi_loc(n_q_points);

  // name the different components of the shape functions
  const FEValuesExtractors::Scalar u (0);
  const FEValuesExtractors::Scalar v (1);

  typename DoFHandler<dim,spacedim>::active_cell_iterator cell;
  
  // loop over all cells
  for (cell=begin; cell!=end; ++cell)
    {
      // reset values each loop
      cell_matrix = 0;
      fe_values.reinit (cell);

      // get values of old u and v
      fe_values[u].get_function_values (xi, u_loc);
      fe_values[v].get_function_values (xi, v_loc);
      
      // loop over quadrature points
      for (unsigned int q_point=0; q_point<n_q_points; ++q_point)
         {	
		  // query Jacobian and weight only once
		  double JxW = fe_values.JxW(q_point);
		  
          for (unsigned int i=0; i<dofs_per_cell; ++i)
          {
          const double phi_i_u = fe_values[u].value (i, q_point);
          const double phi_i_v = fe_values[v].value (i, q_point);

 	  for (unsigned int j=0; j<dofs_per_cell; ++j)
            { 
            const double phi_j_u = fe_values[u].value (j, q_point);
            const double phi_j_v = fe_values[v].value (j, q_point);

                 // assemble 'non linear' LHS matrices
               cell_matrix(i,j) +=  ( ( u_loc[q_point]*v_loc[q_point]* phi_i_u * phi_j_u 
                                      + u_loc[q_point]*u_loc[q_point]* phi_i_u * phi_j_v 
                                      + v_loc[q_point]*u_loc[q_point]* phi_i_v * phi_j_u 
                                      + u_loc[q_point]*u_loc[q_point]* phi_i_v * phi_j_v 
                                       )*JxW ); 
          
           } //end for dof j
        } //end for dof i
      }  // end for q_point
	  
      // add local contributions to global system
      cell->get_dof_indices (local_dof_indices);

      // get assembler lock since writing to global object
      assembler_lock.acquire ();
      for (unsigned int i=0; i<dofs_per_cell; ++i)
        for (unsigned int j=0; j<dofs_per_cell; ++j)
            B_matrix.add (local_dof_indices[i],
			  local_dof_indices[j],
			  cell_matrix(i,j));
     // finished writing to global object, so release assembler lock
     assembler_lock.release ();
   } //end for cell
  
} 


  ///////////////////////////
 //  Solver newton system //
///////////////////////////

// #solve_system# //

template <int dim, int spacedim>
void RDProblem<dim,spacedim>::solve_system (BlockVector<double> &return_vector) 
{
  SolverControl                      solver_control (10000, 1e-10);
  SolverGMRES<BlockVector<double> >  solver(solver_control);

  //PreconditionJacobi<> preconditioner;
  //preconditioner.initialize(system_matrix,0.6);

  solver.solve (system_matrix, return_vector, system_rhs,
  //        preconditioner);
	    PreconditionIdentity());

 /* std::cout << "   " << solver_control.last_step()
	    << " Iterations needed to obtain convergence."
	    << std::endl; */
}




  ////////////////////
 //  Solver system //
////////////////////

// #solve_system# //

template <int dim, int spacedim>
void RDProblem<dim,spacedim>::solve_block_system_whole (BlockVector<double> &return_vector) 
{
  // use this when solving non symmetric Newton system
  SolverControl                      solver_control (10000, parameters->matrix_solver_tol);
  SolverGMRES<BlockVector<double> >  solver(solver_control);

  PreconditionJacobi<BlockSparseMatrix<double> > preconditioner;
  preconditioner.initialize(system_matrix, 1.0);

  solver.solve (system_matrix, return_vector, system_rhs,
                preconditioner);
            //    PreconditionIdentity());

 /* std::cout << "   " << solver_control.last_step()
	    << " Iterations needed to obtain convergence."
	    << std::endl; */
}

template <int dim, int spacedim>
void RDProblem<dim,spacedim>::solve_block_system_separate (BlockVector<double> &return_vector) 
{
  // use this when block system has block(0,1)=block(1,0)=0
  SolverControl                      solver_control (10000, parameters->matrix_solver_tol);
  SolverCG<Vector<double> >          solver(solver_control);

  solver.solve (system_matrix.block(0,0), return_vector.block(0), system_rhs.block(0),
                  PreconditionIdentity());
  solver.solve (system_matrix.block(1,1), return_vector.block(1), system_rhs.block(1),
                  PreconditionIdentity());

 /* std::cout << "   " << solver_control.last_step()
	    << " Iterations needed to obtain convergence."
	    << std::endl; */
}


  /////////////////////////////////////////////////////
 //  Compute L2 norm of difference of two solutions //
/////////////////////////////////////////////////////

// #compute_L2norm_of_difference# //

template <int dim, int spacedim>
inline void RDProblem<dim,spacedim>::compute_L2norm_of_difference (const BlockVector<double> &solution_1,
                                                                   const BlockVector<double> &solution_2,
                                                                   std::vector<double>       &return_vector)
{
// return vector must have two elements
BlockVector<double>  tmp_diff = solution_1;
tmp_diff -= solution_2;

const ComponentSelectFunction<spacedim> u_mask (0,2), 
                                        v_mask (1,2);
   
Vector<double>  difference_per_cell(triangulation.n_active_cells());

QTrapez<1>     q_trapez;
QIterated<dim> quadrature (q_trapez, (parameters->n_gauss_quad)+2);
//QGauss<spacedim> quadrature(3);

VectorTools::integrate_difference(mapping, dof_handler,
                                  tmp_diff,
                                  ZeroFunction<spacedim>(2),
                                  difference_per_cell,
                                  quadrature,
                                  VectorTools::L2_norm,
                                  &u_mask);

return_vector[0] = difference_per_cell.l2_norm();
difference_per_cell = 0;

VectorTools::integrate_difference(mapping, dof_handler,
                                  tmp_diff,
                                  ZeroFunction<spacedim>(2),
                                  difference_per_cell,
                                  quadrature,
                                  VectorTools::L2_norm,
                                  &v_mask);

return_vector[1] = difference_per_cell.l2_norm();
}


template <int dim, int spacedim>
inline void RDProblem<dim,spacedim>::compute_L2norm_of_difference (const BlockVector<double> &solution_1,
                                                                   const BlockVector<double> &solution_2,
                                                                   double                    &return_value)
{
// return vector must have two elements
BlockVector<double>  tmp_diff = solution_1;
tmp_diff -= solution_2;
   
Vector<double>  difference_per_cell(triangulation.n_active_cells());

QTrapez<1>     q_trapez;
QIterated<dim> quadrature (q_trapez, (parameters->n_gauss_quad)+2);
//QGauss<spacedim> quadrature(3);

VectorTools::integrate_difference(mapping, dof_handler,
                                  tmp_diff,
                                  ZeroFunction<spacedim>(2),
                                  difference_per_cell,
                                  quadrature,
                                  VectorTools::L2_norm);

return_value = difference_per_cell.l2_norm();
}






  ///////////////////////////////
 // Output nonlin convergence //
///////////////////////////////

// #output_solution# //

template <int dim, int spacedim>
inline void RDProblem<dim,spacedim>::output_nonlin_conv (std::string output_name) const
{

std::ofstream output (output_name.c_str(), std::ios::out | std::ios::app ); 

output << std::setprecision(6) << std::scientific
       << total_count << " "
//       << nonlin_L2diff[0] << " " 
//       << nonlin_L2diff[1] << "\n";
       << nonlin_L2diff_xi << std::endl;
output.close();

}

  /////////////////////////////////
 // Output solution convergence //
/////////////////////////////////

// #output_solution# //

template <int dim, int spacedim>
inline void RDProblem<dim,spacedim>::output_solution_conv(std::string output_name) const
{

std::ofstream output (output_name.c_str(), std::ios::out | std::ios::app ); 

output << std::fixed
       << t << " "
       << std::setprecision(6) << std::scientific
       << solution_L2diff[0]/(common_parameters->tau) << " " 
       << solution_L2diff[1]/(common_parameters->tau)  << " "
       << nonlin_count <<  "\n";
output.close();
}

  //////////////////////
 // Output solutions //
//////////////////////

// #output_solution# //

template <int dim, int spacedim>
void RDProblem<dim,spacedim>::output_solution (BlockVector<double> output_solution,
                                      std::string         output_name) 
{
  // Make vector to store names of solutions
  std::vector<std::string> solution_names;

  if(dim==spacedim){
    solution_names.push_back ("u");
    solution_names.push_back ("v");}
  else {
    solution_names.push_back ("r");
    solution_names.push_back ("s");}
  
  DataOut<dim,DoFHandler<dim,spacedim> > data_out;

  data_out.attach_dof_handler(dof_handler);
  data_out.add_data_vector (output_solution, solution_names,
                            DataOut<dim,DoFHandler<dim,spacedim> >::type_dof_data);

  data_out.build_patches (mapping,0);

  // write paravire vtu file 
  std::ofstream output (output_name.c_str()); 
  data_out.write_vtu (output);
  
 
}

  ////////////////////////////////
 // Backward Euler Setup System //
////////////////////////////////

// #backward_euler_setup_systemp# //

template <int dim, int spacedim>
void RDProblem<dim,spacedim>::BE_initialise()
{
  // assemble the parts of rhs and lhs here that don't change for efficieny
  base_lhs=0;
  base_rhs=0;
           
  // add (1/tau)*M xi^n
  mass_matrix.vmult(base_rhs,old_solution);
  base_rhs /= (common_parameters->tau);
   
  // add gamma*a and gamma*b terms
   base_rhs += constant_rhs;
    
  // base lhs is ( [(1/tau + gamma)M + A , 0],[0, (1/tau)M + dA ] )
  // block(n,m) is nth row, mth column
   {
    base_lhs.block(0,0).add((1./(common_parameters->tau)) + (parameters->gamma), mass_matrix.block(0,0)); 
    base_lhs.block(0,0).add(1., laplace_matrix.block(0,0) );
    base_lhs.block(1,1).add(1./(common_parameters->tau), mass_matrix.block(1,1) ); 
    base_lhs.block(1,1).add((parameters->d), laplace_matrix.block(1,1) );
   }

 }
 
template <int dim, int spacedim>
void RDProblem<dim,spacedim>::BE_assemble_newton()
{
 // initialise system and rhs vector to base values
 system_matrix.copy_from(base_lhs);
 system_rhs = base_rhs;  
 lin_solution = solution; 
 get_B_matrix(lin_solution);   
 
  // set up newton system
  // add B contributions to lhs
  system_matrix.block(0,0).add( -2.*(parameters->gamma), B_matrix.block(0,0) );
  system_matrix.block(0,1).add(-(parameters->gamma), B_matrix.block(0,1) );
  system_matrix.block(1,0).add(2.*(parameters->gamma), B_matrix.block(1,0) );
  system_matrix.block(1,1).add((parameters->gamma), B_matrix.block(1,1) );
  // add remaining contributions to rhs
  {
   BlockSparseMatrix<double> tmp_matrix(block_sparsity_pattern);
     tmp_matrix.copy_from(base_lhs);
     tmp_matrix *= (-1.);
      tmp_matrix.vmult_add(system_rhs, lin_solution); 
   }
   {
    SparseMatrix<double> tmp_matrix(block_sparsity_pattern.block(0,0));
       tmp_matrix.copy_from(B_matrix.block(0,0));
       tmp_matrix *= (parameters->gamma);
       tmp_matrix.vmult_add(system_rhs.block(0), lin_solution.block(0)); 
    }
    {
     SparseMatrix<double> tmp_matrix(block_sparsity_pattern.block(1,1));
       tmp_matrix.copy_from(B_matrix.block(1,1));
       tmp_matrix *= -(parameters->gamma);
       tmp_matrix.vmult_add(system_rhs.block(1), lin_solution.block(1)); 
    } 
             
}

  ////////////////////////////
 // Crank -Nicolson System //
////////////////////////////

// #CN_timestep# //

template <int dim, int spacedim>
void RDProblem<dim,spacedim>::CN_timestep(const BlockVector<double> &additional_rhs_terms)
{
      t += (common_parameters->tau);
	  ++time_count;
      old_solution = solution;



         // assemble the parts of rhs and lhs here that don't change for efficieny
      BlockSparseMatrix<double> base_lhs(system_matrix.get_sparsity_pattern());
      BlockVector<double> base_rhs;
      base_rhs.reinit(system_rhs);
      base_lhs=0;
      base_rhs=0;
           
      
      // base lhs is ( [(1/tau + .5*gamma)M + .5*A , 0],[0, (1/tau)M + .5*dA ] )
      // block(n,m) is nth row, mth column

      // add gamma*a and gamma*b terms
      base_rhs += constant_rhs;

      // add the mass and laplace contributions
      {
      BlockSparseMatrix<double> tmp_matrix(block_sparsity_pattern);
      tmp_matrix = 0;
      tmp_matrix.block(0,0).add((1./(common_parameters->tau))+.5*(parameters->gamma), mass_matrix.block(0,0)); 
      tmp_matrix.block(0,0).add(.5, laplace_matrix.block(0,0) );
      tmp_matrix.block(1,1).add((1./(common_parameters->tau)), mass_matrix.block(1,1)); 
      tmp_matrix.block(1,1).add(.5*(parameters->d), laplace_matrix.block(1,1) );

      base_lhs.add(1.,tmp_matrix);
           
      tmp_matrix *= -1.;
      tmp_matrix.add((2./(common_parameters->tau)),mass_matrix);
      tmp_matrix.vmult_add(base_rhs,old_solution);       
      }
       // add B matrix contribution
      {
       BlockSparseMatrix<double> tmp_matrix(block_sparsity_pattern);
       tmp_matrix = 0;
       get_B_matrix(old_solution);

       tmp_matrix.block(0,0).copy_from(B_matrix.block(0,0));
       tmp_matrix.block(1,1).copy_from(B_matrix.block(1,1));

       tmp_matrix *= .5*(parameters->gamma);
       tmp_matrix.block(1,1) *= -1.;
       
       tmp_matrix.vmult_add(base_rhs,old_solution);
      }     

      // add additional input terms
      base_rhs += additional_rhs_terms;

      // nonlin loop
      for( nonlin_count = 1 ;  
           nonlin_count <= (parameters->max_nonlin_iter); 
           ++nonlin_count,++total_count)
         {
           // initialise system and rhs vector to base values
          system_matrix.copy_from(base_lhs);
          system_rhs = base_rhs;
          lin_solution = solution;
          get_B_matrix(lin_solution);        
	 	  
          /*if( (parameters->nonlin_solver)=="Picard")
          {
           // add B contributions to lhs
           system_matrix.block(0,0).add( -.5*(parameters->gamma), B_matrix.block(0,0) );
           system_matrix.block(1,1).add(.5*(parameters->gamma), B_matrix.block(1,1) );
           // solve system
           solve_block_system_separate(solution);
          } // end if picard

          else if( (parameters->nonlin_solver)=="Newton")*/
          {
           BlockVector<double> delta;
           delta.reinit(solution);
           delta=0;
            // set up newton system
            // add B contributions to lhs
           system_matrix.block(0,0).add( -(parameters->gamma), B_matrix.block(0,0) );
           system_matrix.block(0,1).add(-.5*(parameters->gamma), B_matrix.block(0,1) );
           system_matrix.block(1,0).add((parameters->gamma), B_matrix.block(1,0) );
           system_matrix.block(1,1).add(.5*(parameters->gamma), B_matrix.block(1,1) );
            // add remaining contributions to rhs
           {
             BlockSparseMatrix<double> tmp_matrix(block_sparsity_pattern);
             tmp_matrix.copy_from(base_lhs);
             tmp_matrix *= (-1.);
             tmp_matrix.vmult_add(system_rhs, lin_solution); 
           }
           {
            SparseMatrix<double> tmp_matrix(block_sparsity_pattern.block(0,0));
            tmp_matrix.copy_from(B_matrix.block(0,0));
            tmp_matrix *= .5*(parameters->gamma);
            tmp_matrix.vmult_add(system_rhs.block(0), lin_solution.block(0)); 
           }
           {
            SparseMatrix<double> tmp_matrix(block_sparsity_pattern.block(1,1));
            tmp_matrix.copy_from(B_matrix.block(1,1));
            tmp_matrix *= -.5*(parameters->gamma);
            tmp_matrix.vmult_add(system_rhs.block(1), lin_solution.block(1)); 
           } 
          
          solve_block_system_whole(delta);
             // update solution
          solution = 0;
          solution.add(1. ,delta, 1., lin_solution);
          } // end if Newton

          compute_L2norm_of_difference(lin_solution,solution,nonlin_L2diff);
          output_nonlin_conv(output_directory+"/iter.txt");
          
          (*shared_out) << std::setprecision(5) << std::fixed << "t: " << t 
		            << " iter: " << nonlin_count;
          (*shared_out) << std::setprecision(3) << std::scientific         
                    << "  L2 diff u : " << nonlin_L2diff[0]  
                    << "  L2 diff v : " << nonlin_L2diff[1]  << std::endl;
     
        if ((nonlin_L2diff[0] < (parameters->nonlin_solver_tol))
                 && (nonlin_L2diff[1] < (parameters->nonlin_solver_tol)))
           break;
         }
		 
	process_solution();	 
      
}


  ///////////////////
 // FST Time Loop //
///////////////////

// #fractional_step_theta_timeloop# //

template <int dim, int spacedim>
void RDProblem<dim,spacedim>::FST_timestep_1_assemble()
{
	 lin_solution = solution; //use lin_solution as name for old solution
	 system_matrix = 0;
     system_rhs = 0;

     get_B_matrix(lin_solution);
     	  
     // create lhs matrix
     system_matrix.add(1./((common_parameters->tau)*(common_parameters->theta)),mass_matrix);
     system_matrix.block(0,0).add((parameters->gamma), mass_matrix.block(0,0) );
     system_matrix.block(0,0).add(1., laplace_matrix.block(0,0) );
     system_matrix.block(1,1).add((parameters->d), laplace_matrix.block(1,1) );

      // create rhs
         // add term from time derivative
      mass_matrix.vmult(system_rhs,lin_solution);
      system_rhs /= ((common_parameters->theta)*(common_parameters->tau));
       
        // add constant term
      system_rhs += constant_rhs;

       // add B terms
      {
        SparseMatrix<double> tmp_matrix(block_sparsity_pattern.block(0,0));
            tmp_matrix.copy_from(B_matrix.block(0,0));
            tmp_matrix *= (parameters->gamma);
            tmp_matrix.vmult_add(system_rhs.block(0), lin_solution.block(0)); 
      }
      {
        SparseMatrix<double> tmp_matrix(block_sparsity_pattern.block(1,1));
            tmp_matrix.copy_from(B_matrix.block(1,1));
            tmp_matrix *= -(parameters->gamma);
            tmp_matrix.vmult_add(system_rhs.block(1), lin_solution.block(1)); 
      }
} 

template <int dim, int spacedim>
void RDProblem<dim,spacedim>::FST_timestep_2_initialise()
{ // get the base lhs and base rhs for the newton loop
    // assemble the parts of rhs and lhs here that don't change for efficieny
      lin_solution = solution;
      base_lhs=0; 
      base_rhs=0;
     
      // base lhs is (1/(1-2*theta))M
      base_lhs.copy_from(mass_matrix);
      base_lhs /= ((1.-2.*(common_parameters->theta))*(common_parameters->tau)); 

      // create base rhs 
       // add (time derivative contribution
      base_lhs.vmult(base_rhs,solution);
   
      // add gamma*a and gamma*b terms
      base_rhs += constant_rhs;
   
     // add mass and laplace matrix contributions
      {
        SparseMatrix<double> tmp_matrix(block_sparsity_pattern.block(0,0));
            tmp_matrix.copy_from(mass_matrix.block(0,0));
            tmp_matrix *= -(parameters->gamma);
            tmp_matrix.vmult_add(base_rhs.block(0), solution.block(0)); 
      }
      {
        SparseMatrix<double> tmp_matrix(block_sparsity_pattern.block(0,0));
            tmp_matrix.copy_from(laplace_matrix.block(0,0));
            tmp_matrix *= -1.;
            tmp_matrix.vmult_add(base_rhs.block(0), solution.block(0)); 
      }
      {
        SparseMatrix<double> tmp_matrix(block_sparsity_pattern.block(1,1));
            tmp_matrix.copy_from(laplace_matrix.block(1,1));
            tmp_matrix *= -(parameters->d);
            tmp_matrix.vmult_add(base_rhs.block(1), solution.block(1)); 
      }


}

template <int dim, int spacedim>
void RDProblem<dim,spacedim>::FST_timestep_2_assemble_newton()
{ 
   // base lhs and base rhs already setup 
           // initialise system and rhs vector to base values
   system_matrix.copy_from(base_lhs);
   system_rhs = base_rhs;
   lin_solution = solution;
   get_B_matrix(lin_solution);        

   // set up newton system
   // add B contributions to lhs
   system_matrix.block(0,0).add( -2.*(parameters->gamma), B_matrix.block(0,0) );
   system_matrix.block(0,1).add(-(parameters->gamma), B_matrix.block(0,1) );
   system_matrix.block(1,0).add(2.*(parameters->gamma), B_matrix.block(1,0) );
   system_matrix.block(1,1).add((parameters->gamma), B_matrix.block(1,1) );

   // add remaining contributions to rhs
   {
    BlockSparseMatrix<double> tmp_matrix(system_matrix.get_sparsity_pattern());
    tmp_matrix.copy_from(base_lhs);
    tmp_matrix *= -1.;
    tmp_matrix.vmult_add(system_rhs, lin_solution); 
   }
   {
    SparseMatrix<double> tmp_matrix(block_sparsity_pattern.block(0,0));
    tmp_matrix.copy_from(B_matrix.block(0,0));
    tmp_matrix *= (parameters->gamma);
    tmp_matrix.vmult_add(system_rhs.block(0), lin_solution.block(0)); 
   }
   {
    SparseMatrix<double> tmp_matrix(block_sparsity_pattern.block(1,1));
    tmp_matrix.copy_from(B_matrix.block(1,1));
    tmp_matrix *= -(parameters->gamma);
    tmp_matrix.vmult_add(system_rhs.block(1), lin_solution.block(1)); 
   } 
}





template <int dim, int spacedim>
void RDProblem<dim,spacedim>::FST_timestep()
{ 
   /*t += (common_parameters->tau);
   ++time_count;
   old_solution = solution;
    FST_timestep_1(additional_rhs_terms);
    FST_timestep_2(additional_rhs_terms);
    FST_timestep_1(additional_rhs_terms);
    process_solution(); */
}

template <int dim, int spacedim>
void RDProblem<dim,spacedim>::update_time()
{ 
   t += (common_parameters->tau);
   ++time_count;
   old_solution = solution;
}
	    
template <int dim, int spacedim>
void RDProblem<dim,spacedim>::process_solution()
{
   //// now process solutions as necessary
   compute_L2norm_of_difference(old_solution,solution,solution_L2diff); 
   output_solution_conv(output_directory + "/conv.txt");
	 
      if (common_parameters->save_animation)
     {
	 // output result every 100 timesteps
	 if ( (time_count % 100 == 0)  )
	 {
	   std::stringstream out;
           out << output_directory << "/" << common_parameters->testID << "-solution_" << time_count << ".vtu";
           output_solution(solution,out.str());  
	  }
     }
     else
       // output result every 100 timesteps
       if ( (time_count % 100 == 0)  )
         {
           std::stringstream out;
           out << output_directory << "/" << common_parameters->testID << "-solution.vtu";
           output_solution(solution,out.str());  
         }
}


   ///////////////////////////////////////////
  //  Class to handler the coupled system  //
 ///////////////////////////////////////////

template<int dim,int spacedim>
class CoupledSystemHandler
{
  public:
  CoupledSystemHandler(CommonParameters &common_parameterss,
                       Parameters       &bulk_parameterss,
                       Parameters       &surf_parameterss);
  void run();

  private:
  inline void        add_to_matrix(const double a,const  BlockSparseMatrix<double> &M_1,const int i,const int j,
                                                         BlockSparseMatrix<double> &M_2,const int k,const int l);
  inline void  initialise();
  void         get_cell_map(); 
  void         setup_full_system();
  inline void  get_base_lhs();
  void         get_base_lhs_part(const typename DoFHandler<spacedim>::active_cell_iterator     &begin_bulk,
                                 const typename DoFHandler<spacedim>::active_cell_iterator     &end_bulk,
                                 const typename DoFHandler<dim,spacedim>::active_cell_iterator &begin_surf,
                                 const typename DoFHandler<dim,spacedim>::active_cell_iterator &end_surf);
  inline void get_surf_coupling_term();
  void        get_surf_coupling_term_part(const typename DoFHandler<dim,spacedim>::active_cell_iterator &begin,
                                          const typename DoFHandler<dim,spacedim>::active_cell_iterator &end);
  inline void get_bulk_coupling_term();
  void        get_bulk_coupling_term_part(const typename DoFHandler<spacedim>::active_cell_iterator &begin,
                                          const typename DoFHandler<spacedim>::active_cell_iterator &end);
  void BE_timestep();
  void FST_timestep_1();
  void FST_timestep_2();                                    
  void create_output_directory();
  void output_solution();
  void solve_system (BlockVector<double> &return_vector) ;
  void compute_L2norm_of_difference (const   BlockVector<double> &solution_1,
                                     const   BlockVector<double> &solution_2,
                                     double  &return_value);

  std::ostream      shared_out;     
  CommonParameters  *common_parameters;
  Parameters        *bulk_parameters,
                    *surf_parameters;
  RDProblem<spacedim,spacedim> RD_bulk;
  RDProblem<dim,spacedim>      RD_surf;
  std::string       output_directory, 
                    bulk_output_directory,
                    surf_output_directory;
 //
  BlockSparsityPattern      block_sparsity_pattern;
  BlockSparseMatrix<double> system_matrix,base_lhs;
  BlockVector<double>       coupling_term_bulk, coupling_term_surf, 
                            system_rhs, solution, 
                            lin_solution, old_solution;
   double t;

  // object to synchronise threads
  Threads::Mutex     assembler_lock;

  std::map<typename DoFHandler<dim,spacedim>::active_cell_iterator,
           typename DoFHandler<spacedim>::active_cell_iterator >      surf_cell_to_bulk_cell_map;
  std::map<typename DoFHandler<spacedim>::active_cell_iterator,
           typename DoFHandler<dim,spacedim>::active_cell_iterator >  bulk_cell_to_surf_cell_map;
  std::map<typename DoFHandler<spacedim>::active_cell_iterator,
           std::vector<Quadrature<dim> > >                            bulk_cell_to_surf_quad_map;
  std::map<typename DoFHandler<dim,spacedim>::active_cell_iterator,
           std::vector<Quadrature<spacedim> > >                       surf_cell_to_bulk_quad_map;

};


   ///////////////////
  //  Constructor  //
 ///////////////////
template<int dim,int spacedim>
CoupledSystemHandler<dim,spacedim>::CoupledSystemHandler(CommonParameters &common_parameterss,
                                                         Parameters       &bulk_parameterss,
                                                         Parameters       &surf_parameterss):
 shared_out(std::cout.rdbuf()),
 common_parameters(&common_parameterss),
 bulk_parameters(&bulk_parameterss),
 surf_parameters(&surf_parameterss),
 RD_bulk(shared_out,
         bulk_parameterss,
         common_parameterss),
 RD_surf(shared_out,
         surf_parameterss,
         common_parameterss)
{
}

   //////////////////
  //  Initialise  //
 //////////////////

template<int dim,int spacedim>
void CoupledSystemHandler<dim,spacedim>::initialise()
{
	  // create output directory
  create_output_directory();
  bulk_output_directory = output_directory + "/bulk",
  surf_output_directory = output_directory + "/surf";
  RD_bulk.output_directory = bulk_output_directory;
  RD_surf.output_directory = surf_output_directory;

 // initialise the rd problems	
 RD_bulk.initialise();
 RD_surf.initialise();		

  // initialise coupling terms
  coupling_term_bulk.reinit(RD_bulk.solution);
  coupling_term_surf.reinit(RD_surf.solution);

  // set up system for solving
  get_cell_map();

  shared_out << "Setting up matrices and vectors..." << std::endl;
  setup_full_system();
  get_base_lhs();
  
 // base_lhs.print_formatted	(std::cout,3,true,0," ");
 
  // initialise solution
   solution.block(0) = RD_bulk.solution.block(0);
     solution.block(1) = RD_bulk.solution.block(1);
   solution.block(2) = RD_surf.solution.block(0);
     solution.block(3) = RD_surf.solution.block(1);


}

   //////////////////////////////
  //  Create Ouput Directory  //
 //////////////////////////////

template<int dim,int spacedim>
void CoupledSystemHandler<dim,spacedim>::create_output_directory()
{

 // create output directory
  output_directory = (common_parameters -> testID) + "-results";
 
 // check if output directory already exists - check with user before deleting
    if ( access( output_directory.c_str(), 0 ) == 0 )
   {
      struct stat status;
      stat( output_directory.c_str(), &status );

      if ( status.st_mode & S_IFDIR )
      {
        shared_out << "The directory \"" << output_directory
                   << "\" already exists. OK to delete directory and its contents? [y/n]" << std::endl;
		std::string go_ahead;		   
		 while (go_ahead != "y")
		 {
            std::cin >> go_ahead;
			if (go_ahead == "n")
			{
            std::printf("Goodbye!\n");	
            abort();			
			}
	    }
		
		std::string tmp= "rm -rf "+ output_directory;
		system(tmp.c_str());
		mkdir(output_directory.c_str(),0777);      std::string tmp_string;
                 tmp_string = output_directory + "/bulk";
                mkdir(tmp_string.c_str(),0777);
                 tmp_string = output_directory + "/surf";
                mkdir(tmp_string.c_str(),0777);
      }
   }
   else
   {
      shared_out << "Creating output directory \"" << output_directory << "\"" << std::endl;
	  mkdir(output_directory.c_str(),0777);
      std::string tmp_string;
     tmp_string = output_directory + "/bulk";
          mkdir(tmp_string.c_str(),0777);
     tmp_string = output_directory + "/surf";
          mkdir(tmp_string.c_str(),0777);
   } 

}

  ///////////////////
 // Add to matrix //
///////////////////

// add a*(M_1).block(i,j) to (M_2).block(k,l)
template<int dim,int spacedim>
void CoupledSystemHandler<dim,spacedim>::add_to_matrix(const double a,const  BlockSparseMatrix<double> &M_1,const int i,const int j,
                                                       BlockSparseMatrix<double> &M_2,const int k,const int l)
{

 FullMatrix<double>   *tmp_full_matrix   = new FullMatrix<double>();
 SparseMatrix<double> *tmp_sparse_matrix = new SparseMatrix<double>(block_sparsity_pattern.block(k,l));
      // get entries to add in in a full matrix
   tmp_full_matrix->copy_from(M_1.block(i,j));   
     // now copy entries from full to sparse
   tmp_sparse_matrix->copy_from(*tmp_full_matrix);
     // and add things together now 
 assembler_lock.acquire ();    
 M_2.block(k,l).add(a, *tmp_sparse_matrix);
 assembler_lock.release();
 delete tmp_full_matrix;
 delete tmp_sparse_matrix;

}


  //////////////////
 // Get cell map //
//////////////////


template<int dim,int spacedim>
void CoupledSystemHandler<dim,spacedim>::get_cell_map()
{
   typedef typename DoFHandler<dim,spacedim>::active_cell_iterator active_cell_iterator_surf;
   typedef typename DoFHandler<spacedim>::active_cell_iterator     active_cell_iterator_bulk; 
                                      
   active_cell_iterator_surf cell_surf = RD_surf.dof_handler.begin_active();
   active_cell_iterator_bulk cell_bulk = RD_bulk.dof_handler.begin_active();


  // get a list of cells in the bulk that are on the bounadary
  std::vector<active_cell_iterator_bulk>* boundary_cells_bulk =      new() std::vector<active_cell_iterator_bulk>;
  std::vector<active_cell_iterator_bulk>* boundary_cells_bulk_copy = new() std::vector<active_cell_iterator_bulk>;

  // loop over bulk cells to find boundar cells
  for( ;cell_bulk!=RD_bulk.dof_handler.end() ; ++cell_bulk)
    if(cell_bulk->at_boundary())
      {
        boundary_cells_bulk->push_back(cell_bulk);    
        boundary_cells_bulk_copy->push_back(cell_bulk);  
      }

std::cout << "Found " << boundary_cells_bulk->size() << " bulk boundary cells." << std::endl;

  // loop over surf cells
  for( ;cell_surf!=RD_surf.dof_handler.end() ; ++cell_surf)
  {
    // get surf cell centre
    Point<spacedim> surf_centre(cell_surf->center());
    
    bool cell_found = false;
//std::cout << "--------------------------" << std::endl;
      // loop over bulk boundary cells
      for(unsigned int i=0 ;i<boundary_cells_bulk_copy->size(); ++i)
      {
        for(unsigned int f=0;f<GeometryInfo<spacedim>::faces_per_cell;++f)
        {
//std::cout << "Comparing surf point " << surf_centre
//    	  << " with bulk point " << 	(*boundary_cells_bulk_copy)[i]->face(f)->center()
// 		  << "...";
			
          if(surf_centre.distance(  ((*boundary_cells_bulk_copy)[i])->face(f)->center())<1.e-7)
           {
//std::cout << "Success." << std::endl;
			 cell_found = true;
             surf_cell_to_bulk_cell_map[cell_surf]=(*boundary_cells_bulk_copy)[i];
             // remove cell from list so we don't have to search through it again
             //boundary_cells_bulk_copy -> erase(boundary_cells_bulk_copy->begin()+i);
             break;
           }
//           else
//std::cout << "Failure." << std::endl;           
		}
         if(cell_found)
            break;		
       }

    if(!cell_found)
       shared_out << "Warning: cell not found for map." << std::endl; 
  } // end for surf cell




  // now make inverse map
  {
    typedef typename std::map<typename DoFHandler<dim,spacedim>::active_cell_iterator,
                              typename DoFHandler<spacedim>::active_cell_iterator >::iterator it_type;   

   for(it_type map_it=surf_cell_to_bulk_cell_map.begin();map_it!=surf_cell_to_bulk_cell_map.end();++map_it)
       bulk_cell_to_surf_cell_map[map_it->second]=(map_it->first);
  }

  // now find projections of quadrature points on unit cells
  // first bulk...
 {
   // define quadrature rule for 2d shape functions
  QGauss<spacedim-1>      face_quadrature_formula(bulk_parameters->n_gauss_quad); 

  // create objects that hold information about 2d shape functions
  FEFaceValues<spacedim> fe_face_values (RD_bulk.mapping, RD_bulk.fe, face_quadrature_formula, 
                                         update_quadrature_points);
  // useful synonyms 
  const unsigned int                 n_q_points        = face_quadrature_formula.size();
  std::vector<Point<spacedim> >      q_points(n_q_points);

  active_cell_iterator_bulk cell_bulk = RD_bulk.dof_handler.begin_active();
  
  // loop thorugh bulk boundary cells
  for(unsigned int i=0;i< boundary_cells_bulk->size();++i)
   for(unsigned int f=0; f<GeometryInfo<spacedim>::faces_per_cell;++f)
    if( ((*boundary_cells_bulk)[i])->face(f)->at_boundary())
    {
     // now we are on surface
     fe_face_values.reinit((*boundary_cells_bulk)[i],f);

      q_points = fe_face_values.get_quadrature_points();
      // transform q points to surface unit cells and put into vector
      std::vector<Quadrature<dim> > local_unit_quads(n_q_points);
         for (unsigned int q=0; q<n_q_points; ++q)
          {

           const Point<dim> unit_q_point 
                      = RD_surf.mapping.transform_real_to_unit_cell(
                              bulk_cell_to_surf_cell_map[(*boundary_cells_bulk)[i]],q_points[q]);  

           const Quadrature<dim> tmp_quadrature(unit_q_point);
           local_unit_quads[q] = tmp_quadrature;           
           }         	
       bulk_cell_to_surf_quad_map[(*boundary_cells_bulk)[i]]=local_unit_quads;       
      } // end for face
	
   } // end for bulk

  // ...now the surface 
 {
  // create objects that hold information about 2d shape functions
  FEValues<dim,spacedim> fe_values (RD_surf.mapping, RD_surf.fe, RD_surf.quadrature_formula, 
                                         update_quadrature_points);
  // useful synonyms 
  const unsigned int                 n_q_points = RD_surf.quadrature_formula.size();
  std::vector<Point<spacedim> >      q_points(n_q_points);

  active_cell_iterator_surf cell_surf = RD_surf.dof_handler.begin_active();

  // loop thorugh bulk boundary cells
  for(; cell_surf!=RD_surf.dof_handler.end();++cell_surf)
    {
     fe_values.reinit(cell_surf);

      q_points = fe_values.get_quadrature_points();
      // transform q points to surface unit cells and put into vector
      std::vector<Quadrature<spacedim> > local_unit_quads(n_q_points);
         for (unsigned int q=0; q<n_q_points; ++q)
          {
           const Point<spacedim> unit_q_point 
                      = RD_bulk.mapping.transform_real_to_unit_cell(
                              surf_cell_to_bulk_cell_map[cell_surf],q_points[q]);  
           const Quadrature<spacedim> tmp_quadrature(unit_q_point); 
           local_unit_quads[q] = tmp_quadrature;
           }

       surf_cell_to_bulk_quad_map[cell_surf]=local_unit_quads;       
	
   } // end for surf cells
  } // end for surf


  // delete uneccesary vector now 
  delete boundary_cells_bulk_copy;
  delete boundary_cells_bulk;

}

  ///////////////////////
 // Setup full system //
///////////////////////


template<int dim,int spacedim>
void CoupledSystemHandler<dim,spacedim>::setup_full_system()
{
   // block sparse matrix has size (row x col)
   //              n_u         n_v         n_r         n_s
   //              0           1           2           3
   // n_u  0  (n_u x n_u) (n_u x n_v) (n_u x n_r) (n_u x n_s)
   // n_v  1  (n_v x n_u) (n_v x n_v) (n_v x n_r) (n_v x n_s)
   // n_r  2  (n_r x n_u) (n_r x n_v) (n_r x n_r) (n_r x n_s) 
   // n_s  3  (n_s x n_u) (n_s x n_v) (n_s x n_r) (n_s x n_s) 


  const unsigned int n_u = RD_bulk.n_u,
                     n_v = RD_bulk.n_v,
                     n_r = RD_surf.n_u, 
                     n_s = RD_surf.n_v;
  const unsigned int n_bulk  = n_u+n_v,
                     n_surf  = n_r+n_s,
                     n_total = n_bulk+n_surf;   
         
  // do the same for the actual block sparsity pattern
 // now get the rest of the sparsity pattern
shared_out << "Builing initial sparsity pattern..." << std::endl;
  block_sparsity_pattern.reinit(4,4);
     // copy sparsity patterns from RDbulk   
   
  block_sparsity_pattern.block(0,0).copy_from(RD_bulk.block_sparsity_pattern.block(0,0));
  block_sparsity_pattern.block(0,1).copy_from(RD_bulk.block_sparsity_pattern.block(0,1));
  block_sparsity_pattern.block(1,0).copy_from(RD_bulk.block_sparsity_pattern.block(1,0));
  block_sparsity_pattern.block(1,1).copy_from(RD_bulk.block_sparsity_pattern.block(1,1));
    // copy sparsity patterns from RDsurf  
  block_sparsity_pattern.block(2,2).copy_from(RD_surf.block_sparsity_pattern.block(0,0));
  block_sparsity_pattern.block(2,3).copy_from(RD_surf.block_sparsity_pattern.block(0,1));
  block_sparsity_pattern.block(3,2).copy_from(RD_surf.block_sparsity_pattern.block(1,0));
  block_sparsity_pattern.block(3,3).copy_from(RD_surf.block_sparsity_pattern.block(1,1)); 

  block_sparsity_pattern.block(0,2).reinit(n_u,n_r,n_r);
  block_sparsity_pattern.block(0,3).reinit(n_u,n_s,n_s);
  block_sparsity_pattern.block(1,2).reinit(n_v,n_r,n_r);
  block_sparsity_pattern.block(1,3).reinit(n_v,n_s,n_s);
   
  block_sparsity_pattern.block(2,0).reinit(n_r,n_u,n_u);
  block_sparsity_pattern.block(2,1).reinit(n_r,n_v,n_v);
  block_sparsity_pattern.block(3,0).reinit(n_s,n_u,n_u);
  block_sparsity_pattern.block(3,1).reinit(n_s,n_v,n_v);
  block_sparsity_pattern.collect_sizes();

  CompressedSimpleSparsityPattern compressed_sparsity_pattern_02,
                            compressed_sparsity_pattern_03,
                            compressed_sparsity_pattern_12, 
                            compressed_sparsity_pattern_13,  
                            compressed_sparsity_pattern_20,
                            compressed_sparsity_pattern_30,
                            compressed_sparsity_pattern_21, 
                            compressed_sparsity_pattern_31; 

  //CompressedSparsityPattern compressed_simple_sparsity_pattern_02,
//				  compressed_simple_sparsity_pattern_03,
//                                compressed_simple_sparsity_pattern_12,
//				  compressed_simple_sparsity_pattern_13,
//                                  compressed_simple_sparsity_pattern_20, 
//				  compressed_simple_sparsity_pattern_30,
//                                  compressed_simple_sparsity_pattern_21,
//				  compressed_simple_sparsity_pattern_31;
                                  

   compressed_sparsity_pattern_02.reinit(n_u,n_r);
   compressed_sparsity_pattern_03.reinit(n_u,n_s);
   compressed_sparsity_pattern_12.reinit(n_v,n_r);
   compressed_sparsity_pattern_13.reinit(n_v,n_s);
   
   compressed_sparsity_pattern_20.reinit(n_r,n_u);
   compressed_sparsity_pattern_21.reinit(n_r,n_v);
   compressed_sparsity_pattern_30.reinit(n_s,n_u);
   compressed_sparsity_pattern_31.reinit(n_s,n_v);   

  for(unsigned int i=0;i<n_u;++i) // row                  
    for(unsigned int j=0;j<n_r;++j) // column
      {
       compressed_sparsity_pattern_02.add(i,j);
       compressed_sparsity_pattern_20.add(j,i);       
      }
  for(unsigned int i=0;i<n_u;++i) // row                  
    for(unsigned int j=0;j<n_s;++j) // column
      {
       compressed_sparsity_pattern_03.add(i,j);
       compressed_sparsity_pattern_30.add(j,i);       
      }
  for(unsigned int i=0;i<n_v;++i) // row                  
    for(unsigned int j=0;j<n_r;++j) // column
      {
       compressed_sparsity_pattern_12.add(i,j);
       compressed_sparsity_pattern_21.add(j,i);       
      }
  for(unsigned int i=0;i<n_v;++i) // row                  
    for(unsigned int j=0;j<n_s;++j) // column
      {
       compressed_sparsity_pattern_13.add(i,j);
       compressed_sparsity_pattern_31.add(j,i);       
      }

  block_sparsity_pattern.block(0,2).copy_from(compressed_sparsity_pattern_02);
  block_sparsity_pattern.block(0,3).copy_from(compressed_sparsity_pattern_03);
  block_sparsity_pattern.block(1,2).copy_from(compressed_sparsity_pattern_12);
  block_sparsity_pattern.block(1,3).copy_from(compressed_sparsity_pattern_13);
   
  block_sparsity_pattern.block(2,0).copy_from(compressed_sparsity_pattern_20);
  block_sparsity_pattern.block(2,1).copy_from(compressed_sparsity_pattern_21);
  block_sparsity_pattern.block(3,0).copy_from(compressed_sparsity_pattern_30);
  block_sparsity_pattern.block(3,1).copy_from(compressed_sparsity_pattern_31);

/*
     // add entries on top left and bottom right   
here("adding to block sparsity pattern");                                 
  for(unsigned int i=0;i<n_bulk;++i) // row                  
    for(unsigned int j=n_bulk;j<n_total;++j) // column
      {
       compressed_block_sparsity_pattern.add(i,j);
       compressed_block_sparsity_pattern.add(j,i);       
      }
*/
here("compressing block sparsity pattern");
  block_sparsity_pattern.compress();
  std::cout << "First block sparsity pattern size = "
          << block_sparsity_pattern.memory_consumption()/1048576. 
          << " MB" << std::endl   ; 

shared_out << "Retrieving example matrix..." << std::endl;   
  base_lhs.reinit(block_sparsity_pattern);
  get_base_lhs();

   // now modify tmp_block_sparsity_pattern to include the more realistic sparsity pattern
   // for the bulk-surface coupling entries
  {
  FullMatrix<double> tmp_matrix;
shared_out << "Correcting initial sparsity pattern..." << std::endl;  
  tmp_matrix.copy_from(base_lhs.block(0,2));
       block_sparsity_pattern.block(0,2).copy_from(tmp_matrix);
  tmp_matrix.copy_from(base_lhs.block(0,3));
       block_sparsity_pattern.block(0,3).copy_from(tmp_matrix);
  tmp_matrix.copy_from(base_lhs.block(1,2));
       block_sparsity_pattern.block(1,2).copy_from(tmp_matrix);
  tmp_matrix.copy_from(base_lhs.block(1,3));
       block_sparsity_pattern.block(1,3).copy_from(tmp_matrix);

  tmp_matrix.copy_from(base_lhs.block(2,0));
       block_sparsity_pattern.block(2,0).copy_from(tmp_matrix);
  tmp_matrix.copy_from(base_lhs.block(2,1));
       block_sparsity_pattern.block(2,1).copy_from(tmp_matrix);
  tmp_matrix.copy_from(base_lhs.block(3,0));
       block_sparsity_pattern.block(3,0).copy_from(tmp_matrix);
  tmp_matrix.copy_from(base_lhs.block(3,1));
       block_sparsity_pattern.block(3,1).copy_from(tmp_matrix);
  }
   // now free base lhs of its sparsity pattern
  base_lhs.clear();

  block_sparsity_pattern.collect_sizes();
  block_sparsity_pattern.compress();
    
std::cout << "Final block sparsity pattern size = "
          << block_sparsity_pattern.memory_consumption()/1048576. 
          << " MB" << std::endl ;

  base_lhs.reinit(block_sparsity_pattern);
  system_matrix.reinit(block_sparsity_pattern);
 /*
{
  std::ofstream output ("sparsity-00.gpl"); 
  block_sparsity_pattern.block(0,0).print_gnuplot(output);
  output.close();
}
{
  std::ofstream output ("sparsity-01.gpl"); 
  block_sparsity_pattern.block(0,1).print_gnuplot(output);
  output.close();
}
{
  std::ofstream output ("sparsity-02.gpl"); 
  block_sparsity_pattern.block(0,2).print_gnuplot(output);
  output.close();
}
{
  std::ofstream output ("sparsity-03.gpl"); 
  block_sparsity_pattern.block(0,3).print_gnuplot(output);
  output.close();
}
{
  std::ofstream output ("sparsity-10.gpl"); 
  block_sparsity_pattern.block(1,0).print_gnuplot(output);
  output.close();
}
{
  std::ofstream output ("sparsity-11.gpl"); 
  block_sparsity_pattern.block(1,1).print_gnuplot(output);
  output.close();
}
{
  std::ofstream output ("sparsity-12.gpl"); 
  block_sparsity_pattern.block(1,2).print_gnuplot(output);
  output.close();
}
{
  std::ofstream output ("sparsity-13.gpl"); 
  block_sparsity_pattern.block(1,3).print_gnuplot(output);
  output.close();
}
{
  std::ofstream output ("sparsity-20.gpl"); 
  block_sparsity_pattern.block(2,0).print_gnuplot(output);
  output.close();
}
{
  std::ofstream output ("sparsity-21.gpl"); 
  block_sparsity_pattern.block(2,1).print_gnuplot(output);
  output.close();
}
{
  std::ofstream output ("sparsity-22.gpl"); 
  block_sparsity_pattern.block(2,2).print_gnuplot(output);
  output.close();
}
{
  std::ofstream output ("sparsity-23.gpl"); 
  block_sparsity_pattern.block(2,3).print_gnuplot(output);
  output.close();
}
{
  std::ofstream output ("sparsity-30.gpl"); 
  block_sparsity_pattern.block(3,0).print_gnuplot(output);
  output.close();
}
{
  std::ofstream output ("sparsity-31.gpl"); 
  block_sparsity_pattern.block(3,1).print_gnuplot(output);
  output.close();
}
{
  std::ofstream output ("sparsity-32.gpl"); 
  block_sparsity_pattern.block(3,2).print_gnuplot(output);
  output.close();
}
{
  std::ofstream output ("sparsity-33.gpl"); 
  block_sparsity_pattern.block(3,3).print_gnuplot(output);
  output.close();
}

*/



  system_rhs.reinit(4);
  system_rhs.block(0).reinit(n_u);
  system_rhs.block(1).reinit(n_v);
  system_rhs.block(2).reinit(n_r);
  system_rhs.block(3).reinit(n_s);  
  system_rhs.collect_sizes();
  
  solution.reinit(4);
  solution.block(0).reinit(n_u);
  solution.block(1).reinit(n_v);
  solution.block(2).reinit(n_r);
  solution.block(3).reinit(n_s);  
  solution.collect_sizes();
  
  lin_solution.reinit(4);
  lin_solution.block(0).reinit(n_u);
  lin_solution.block(1).reinit(n_v);
  lin_solution.block(2).reinit(n_r);
  lin_solution.block(3).reinit(n_s);  
  lin_solution.collect_sizes();
  
  old_solution.reinit(4);
  old_solution.block(0).reinit(n_u);
  old_solution.block(1).reinit(n_v);
  old_solution.block(2).reinit(n_r);
  old_solution.block(3).reinit(n_s);  
  old_solution.collect_sizes();
  
}




  //////////////////
 // Get vase lhs //
//////////////////

template<int dim,int spacedim>
void CoupledSystemHandler<dim,spacedim>::get_base_lhs()
{
 base_lhs = 0;
 
 Threads::ThreadGroup<> threads;
 typedef typename DoFHandler<spacedim>::active_cell_iterator active_cell_iterator_bulk;
 std::vector<std::pair<active_cell_iterator_bulk,active_cell_iterator_bulk> >
                thread_ranges_bulk = Threads::split_range<active_cell_iterator_bulk> (
                                                     RD_bulk.dof_handler.begin_active (),
                                                     RD_bulk.dof_handler.end (),
                                                     common_parameters->n_threads);
 typedef typename DoFHandler<dim,spacedim>::active_cell_iterator active_cell_iterator_surf;
 std::vector<std::pair<active_cell_iterator_surf,active_cell_iterator_surf> >
                thread_ranges_surf = Threads::split_range<active_cell_iterator_surf> (
                                                     RD_surf.dof_handler.begin_active (),
                                                     RD_surf.dof_handler.end (),
                                                     common_parameters->n_threads);

     for (unsigned int thread=0; thread<common_parameters->n_threads; ++thread)
       threads += Threads::new_thread (&CoupledSystemHandler::get_base_lhs_part,
                                       *this,
                                       thread_ranges_bulk[thread].first,
                                       thread_ranges_bulk[thread].second,
                                       thread_ranges_surf[thread].first,
                                       thread_ranges_surf[thread].second);
  threads.join_all (); 
  
}

template<int dim,int spacedim>
void CoupledSystemHandler<dim,spacedim>::get_base_lhs_part(const typename DoFHandler<spacedim>::active_cell_iterator     &begin_bulk,
                                                           const typename DoFHandler<spacedim>::active_cell_iterator     &end_bulk,
                                                           const typename DoFHandler<dim,spacedim>::active_cell_iterator &begin_surf,
                                                           const typename DoFHandler<dim,spacedim>::active_cell_iterator &end_surf)
{
  // base lhs is the coupling contribution to the Newton system

 /////////////////////////////////////////////////////////////////////
 // fill the top right and bottom left blocks of the base lhs first

 // define quadrature rule for bulk shape functions
  QGauss<spacedim-1>      face_quadrature_formula(bulk_parameters->n_gauss_quad); 

  // create objects that hold information about 2d shape functions
  FEFaceValues<spacedim> fe_face_values_bulk (RD_bulk.mapping,RD_bulk.fe,face_quadrature_formula, 
   			                      update_values | update_quadrature_points | update_JxW_values);

  FEValues<dim,spacedim> fe_values_surf (RD_surf.mapping, RD_surf.fe, RD_surf.quadrature_formula, 
   			                      update_values | update_quadrature_points | update_JxW_values);
  // useful synonyms 
  const unsigned int          dofs_per_cell_bulk  = RD_bulk.fe.dofs_per_cell,
                              dofs_per_cell_surf  = RD_surf.fe.dofs_per_cell;
  const unsigned int          n_q_points_bulk      = face_quadrature_formula.size(),
                              n_q_points_surf      = RD_surf.quadrature_formula.size();
  std::vector<unsigned int>   local_dof_indices_bulk  (dofs_per_cell_bulk),  
                              local_dof_indices_surf  (dofs_per_cell_surf);

  // declare local contributions to global LHS and RHS
  FullMatrix<double>   cell_matrix_bulk_con (dofs_per_cell_bulk,dofs_per_cell_bulk),
                       cell_matrix_bulk_contra (dofs_per_cell_bulk,dofs_per_cell_surf),
                       cell_matrix_surf_con (dofs_per_cell_surf,dofs_per_cell_surf),
                       cell_matrix_surf_contra (dofs_per_cell_surf,dofs_per_cell_bulk);

  // name the different components of the shape functions
  const FEValuesExtractors::Scalar u (0); 
  const FEValuesExtractors::Scalar v (1);
  const FEValuesExtractors::Scalar r (0); 
  const FEValuesExtractors::Scalar s (1);

  typename DoFHandler<spacedim>::active_cell_iterator     cell_bulk;
  typename DoFHandler<dim,spacedim>::active_cell_iterator cell_surf;

 ///////////////////////////////////////////////////
 // first collect terms for the bulk system

 for (cell_bulk = begin_bulk ; cell_bulk!=end_bulk; ++cell_bulk)
  {
   if(cell_bulk->at_boundary() ) // only do something if we are on the boundary
    {
		
	unsigned int face = 0;
    // loop over faces to find which is at boundary 
    for ( ; face < GeometryInfo<spacedim>::faces_per_cell; ++face)
     if ( cell_bulk->face(face)->at_boundary() )
       break;
       
        // reset values each loop
      cell_matrix_bulk_con    = 0;
      cell_matrix_bulk_contra = 0;
         // update surf cell
       fe_face_values_bulk.reinit (cell_bulk,face);
       cell_surf = bulk_cell_to_surf_cell_map[cell_bulk];
       fe_values_surf.reinit(cell_surf);

       for (unsigned int q_point=0; q_point<n_q_points_bulk; ++q_point)
       {	
	    const double JxW = fe_face_values_bulk.JxW(q_point);
          for (unsigned int i=0; i<dofs_per_cell_bulk; ++i)
          {
           const double phi_i_u = fe_face_values_bulk[u].value (i,q_point);
           const double phi_i_v = fe_face_values_bulk[v].value (i,q_point);           
             for (unsigned int j=0; j<dofs_per_cell_bulk; ++j)
             {
              const double phi_j_u = fe_face_values_bulk[u].value (j,q_point);
              const double phi_j_v = fe_face_values_bulk[v].value (j,q_point); 
              cell_matrix_bulk_con(i,j) -= (surf_parameters->gamma)*(
                                               -(common_parameters->beta_1 )*phi_j_u
                                               -(common_parameters->kappa_1)*phi_j_v
                                                )*phi_i_u*JxW ;
              cell_matrix_bulk_con(i,j) -= (surf_parameters->gamma)*(
                                               -(common_parameters->beta_2 )*phi_j_u
                                               -(common_parameters->kappa_2)*phi_j_v
                                                )*phi_i_v*JxW ;      
		     } // end for bulk dof j 
             for (unsigned int j=0; j<dofs_per_cell_surf; ++j)
             {
              const double phi_j_r = fe_values_surf[r].value(j,q_point);
              const double phi_j_s = fe_values_surf[s].value(j,q_point);     
             cell_matrix_bulk_contra(i,j) -= (surf_parameters->gamma)*(common_parameters->alpha_1)*phi_j_r*phi_i_u*JxW ;
             cell_matrix_bulk_contra(i,j) -= (surf_parameters->gamma)*(common_parameters->alpha_2)*phi_j_s*phi_i_v*JxW ;
             } // end for surf dof j
        } //end for bulk dof i
      }  // end for q_point

      // add local contributions to global system
      cell_bulk->get_dof_indices (local_dof_indices_bulk);
      cell_surf->get_dof_indices (local_dof_indices_surf);
    assembler_lock.acquire ();
      for (unsigned int i=0; i<dofs_per_cell_bulk; ++i)
       {
		for (unsigned int j=0; j<dofs_per_cell_bulk; ++j)
     	   base_lhs.add (local_dof_indices_bulk[i],
   	                     local_dof_indices_bulk[j],
 	                     cell_matrix_bulk_con(i,j) ); 
        for (unsigned int j=0; j<dofs_per_cell_surf; ++j)
     	   base_lhs.add (local_dof_indices_bulk[i],
   	                     local_dof_indices_surf[j]+RD_bulk.n_u+RD_bulk.n_v, // add by n_dofs_bulk to shift to correct column
 	                     cell_matrix_bulk_contra(i,j) );
 	    }                   
     // finished writing to global object, so release assembler lock
    assembler_lock.release ();
    } // end if cell on boundary  
   } //end for bulk cell

  ///////////////////////////////////////////////////////////////
  // now for surf coupling terms
 for (cell_surf = begin_surf  ; cell_surf!=end_surf; ++cell_surf)
    {
     // reset values each loop
     cell_matrix_surf_con    = 0; 
     cell_matrix_surf_contra = 0;        
     // update bulk cell
     cell_bulk = surf_cell_to_bulk_cell_map[cell_surf];
     // reinit fevalues
     fe_values_surf.reinit(cell_surf);
    
     for (unsigned int face=0; face < GeometryInfo<spacedim>::faces_per_cell; ++face)
       if ( cell_bulk->face(face)->at_boundary() )
         fe_face_values_bulk.reinit (cell_bulk,face);         
     for (unsigned int q_point=0; q_point<n_q_points_surf; ++q_point)
      {	
	  const double JxW = fe_values_surf.JxW(q_point);
	  for (unsigned int i=0; i<dofs_per_cell_surf; ++i)
       {
        const double phi_i_r = fe_values_surf[r].value (i, q_point);
        const double phi_i_s = fe_values_surf[s].value (i, q_point);
           for (unsigned int j=0; j<dofs_per_cell_surf; ++j)
            {
             const double phi_j_r = fe_values_surf[r].value(j,q_point);
             const double phi_j_s = fe_values_surf[s].value(j,q_point);
             cell_matrix_surf_con(i,j) += (surf_parameters->gamma)*(common_parameters->alpha_1)*phi_j_r*phi_i_r*JxW; 
             cell_matrix_surf_con(i,j) += (surf_parameters->gamma)*(common_parameters->alpha_2)*phi_j_s*phi_i_s*JxW; 
		    }        
           for (unsigned int j=0; j<dofs_per_cell_bulk; ++j)
            {
             const double phi_j_u = fe_face_values_bulk[u].value(j,q_point);
             const double phi_j_v = fe_face_values_bulk[v].value(j,q_point);
             cell_matrix_surf_contra(i,j) += (surf_parameters->gamma)*( -(common_parameters->beta_1 )*phi_j_u 
                                                                         -(common_parameters->kappa_1)*phi_j_v )*phi_i_r*JxW ;
             cell_matrix_surf_contra(i,j) += (surf_parameters->gamma)*( -(common_parameters->beta_2 )*phi_j_u 
                                                                         -(common_parameters->kappa_2)*phi_j_v )*phi_i_s*JxW ;
             } // end for j
         } //end for dof i
      }  // end for q_point
  
      // add local contributions to global system
      cell_bulk->get_dof_indices (local_dof_indices_bulk);
      cell_surf->get_dof_indices (local_dof_indices_surf);
    assembler_lock.acquire ();
      for (unsigned int i=0; i<dofs_per_cell_surf; ++i)
        {
        for (unsigned int j=0; j<dofs_per_cell_surf; ++j)
     	   base_lhs.add (RD_bulk.n_u+RD_bulk.n_v+local_dof_indices_surf[i], 
   	                     RD_bulk.n_u+RD_bulk.n_v+local_dof_indices_surf[j], 
 	                     cell_matrix_surf_con(i,j) ); 			
        for (unsigned int j=0; j<dofs_per_cell_bulk; ++j)
     	   base_lhs.add (local_dof_indices_surf[i]+RD_bulk.n_u+RD_bulk.n_v, // add by n_dofs_bulk to shift to correct row
   	                     local_dof_indices_bulk[j], 
 	                     cell_matrix_surf_contra(i,j) );                
 	                     
 	     }
 	assembler_lock.release();
 }
}

  ////////////////////////////
 // Get bulk coupling term //
////////////////////////////


template<int dim,int spacedim>
void CoupledSystemHandler<dim,spacedim>::get_bulk_coupling_term()
{
 coupling_term_bulk = 0;

 Threads::ThreadGroup<> threads;
 typedef typename DoFHandler<spacedim>::active_cell_iterator active_cell_iterator;
 std::vector<std::pair<active_cell_iterator,active_cell_iterator> >
                thread_ranges = Threads::split_range<active_cell_iterator> (
                                                     RD_bulk.dof_handler.begin_active (),
                                                     RD_bulk.dof_handler.end (),
                                                     common_parameters->n_threads);

     for (unsigned int thread=0; thread<common_parameters->n_threads; ++thread)
       threads += Threads::new_thread (&CoupledSystemHandler::get_bulk_coupling_term_part,
                                       *this,
                                       thread_ranges[thread].first,
                                       thread_ranges[thread].second);
  threads.join_all ();
}


template<int dim,int spacedim>
void CoupledSystemHandler<dim,spacedim>::get_bulk_coupling_term_part(const typename DoFHandler<spacedim>::active_cell_iterator &begin,
                                                                     const typename DoFHandler<spacedim>::active_cell_iterator &end)
{
  // define quadrature rule for 2d shape functions
  QGauss<spacedim-1>      face_quadrature_formula(bulk_parameters->n_gauss_quad); 

  // create objects that hold information about 2d shape functions
  FEFaceValues<spacedim> fe_face_values (RD_bulk.mapping, RD_bulk.fe, face_quadrature_formula, 
   			                 update_values | update_quadrature_points |  update_JxW_values);
  // useful synonyms 
  const unsigned int          dofs_per_cell     = RD_bulk.fe.dofs_per_cell;
  const unsigned int          n_q_points        = face_quadrature_formula.size();
  std::vector<unsigned int>   local_dof_indices  (dofs_per_cell);

  // declare local contributions to global LHS and RHS
  Vector<double>   cell_rhs (dofs_per_cell);

  // store local values of u and v in local vectors
  std::vector<double>            u_loc(n_q_points), v_loc(n_q_points),
                                 r_loc(n_q_points), s_loc(n_q_points);

  std::vector<Vector<double> >   surf_sol_loc(n_q_points,Vector<double>(2)); 


  // name the different components of the shape functions
  const FEValuesExtractors::Scalar u (0);
  const FEValuesExtractors::Scalar v (1);

  typename DoFHandler<spacedim>::active_cell_iterator cell;
  
  // loop over all cells
  for (cell=begin; cell!=end; ++cell)
   if(cell->at_boundary() ) // only do something if we are on the boundary
    {
	  // reset values each loop
      cell_rhs = 0;
      
      // find the face at the boundary
    unsigned int face =0;
      for (; face < GeometryInfo<spacedim>::faces_per_cell; ++face)
         if ( cell->face(face)->at_boundary() )
            break;
          
         fe_face_values.reinit (cell,face);
          // get values of old u and v
          fe_face_values[u].get_function_values (RD_bulk.lin_solution, u_loc);
          fe_face_values[v].get_function_values (RD_bulk.lin_solution, v_loc);

         RD_surf.get_solution_value(RD_surf.lin_solution, surf_sol_loc,
                                    bulk_cell_to_surf_cell_map[cell],bulk_cell_to_surf_quad_map[cell] );

         for (unsigned int q_point=0; q_point<n_q_points; ++q_point)
         {	
	       // query Jacobian and weight only once
	      double JxW = fe_face_values.JxW(q_point);
		  
          for (unsigned int i=0; i<dofs_per_cell; ++i)
           {
            const double phi_i_u = fe_face_values[u].value (i, q_point);
            const double phi_i_v = fe_face_values[v].value (i, q_point);

          cell_rhs(i) +=  (surf_parameters->gamma)*(  (common_parameters->alpha_1)*surf_sol_loc[q_point][0]
                            - (common_parameters->beta_1 )*u_loc[q_point]
                            - (common_parameters->kappa_1)*v_loc[q_point]
                           )*phi_i_u*JxW ;
          cell_rhs(i) +=  (surf_parameters->gamma)*(  (common_parameters->alpha_2)*surf_sol_loc[q_point][1]
                            - (common_parameters->beta_2 )*u_loc[q_point]
                            - (common_parameters->kappa_2)*v_loc[q_point]
                           )*phi_i_v*JxW ;

        } //end for dof i
      }  // end for q_point
      
      // add local contributions to global system
      cell->get_dof_indices (local_dof_indices);
      // get assembler lock since writing to global object
      assembler_lock.acquire ();
      for (unsigned int i=0; i<dofs_per_cell; ++i)
	     coupling_term_bulk(local_dof_indices[i]) += cell_rhs(i);
     // finished writing to global object, so release assembler lock
     assembler_lock.release ();

   } //end for cell
} 

 
  ////////////////////////////
 // Get surf coupling term //
////////////////////////////


template<int dim,int spacedim>
void CoupledSystemHandler<dim,spacedim>::get_surf_coupling_term( )
{
 coupling_term_surf = 0;

 Threads::ThreadGroup<> threads;

 typedef typename DoFHandler<dim,spacedim>::active_cell_iterator active_cell_iterator;
 std::vector<std::pair<active_cell_iterator,active_cell_iterator> >
                thread_ranges = Threads::split_range<active_cell_iterator> (
                                                     RD_surf.dof_handler.begin_active (),
                                                     RD_surf.dof_handler.end (),
                                                     common_parameters->n_threads);
     for (unsigned int thread=0; thread<common_parameters->n_threads; ++thread)
       threads += Threads::new_thread (&CoupledSystemHandler::get_surf_coupling_term_part,
                                       *this,
                                       thread_ranges[thread].first,
                                       thread_ranges[thread].second);
  threads.join_all ();
}


template<int dim,int spacedim>
void CoupledSystemHandler<dim,spacedim>::get_surf_coupling_term_part(const typename DoFHandler<dim,spacedim>::active_cell_iterator &begin,
                                                                     const typename DoFHandler<dim,spacedim>::active_cell_iterator &end )
{
  // create objects that hold information about 2d shape functions
  FEValues<dim,spacedim> fe_values (RD_surf.mapping, RD_surf.fe, RD_surf.quadrature_formula, 
   			            update_values | update_quadrature_points | update_JxW_values);
  // useful synonyms 
  const unsigned int            dofs_per_cell     = RD_surf.fe.dofs_per_cell;
  const unsigned int            n_q_points        = RD_surf.quadrature_formula.size();
  std::vector<unsigned int>     local_dof_indices  (dofs_per_cell);

  // declare local contributions to global LHS and RHS
  Vector<double>   cell_rhs (dofs_per_cell);

  // store local values of u and v in local vectors
  /*Functions::FEFieldFunction<spacedim,
                             DoFHandler<spacedim>,
                             BlockVector<double> > bulk_solution(RD_bulk.dof_handler,RD_bulk.old_solution);*/

 // std::vector<double>  u_loc(n_q_points), v_loc(n_q_points),
  std::vector<double>            r_loc(n_q_points), s_loc(n_q_points);    

  std::vector<Vector<double> >   bulk_loc(n_q_points,Vector<double>(2));

  // name the different components of the shape functions
  const FEValuesExtractors::Scalar r (0);
  const FEValuesExtractors::Scalar s (1);

  typename DoFHandler<dim,spacedim>::active_cell_iterator cell;
  
  // loop over all cells
  for (cell=begin; cell!=end; ++cell)
    {
      // reset values each loop
      cell_rhs = 0;
      fe_values.reinit (cell);

      // get values of old u and v
      fe_values[r].get_function_values (RD_surf.lin_solution, r_loc);
      fe_values[s].get_function_values (RD_surf.lin_solution, s_loc);

      RD_bulk.get_solution_value(RD_bulk.lin_solution,bulk_loc,
                               surf_cell_to_bulk_cell_map[cell],surf_cell_to_bulk_quad_map[cell] );
      // loop over quadrature points
      for (unsigned int q_point=0; q_point<n_q_points; ++q_point)
         {	
	    // query Jacobian and weight only once
	    double JxW = fe_values.JxW(q_point);
		  
          for (unsigned int i=0; i<dofs_per_cell; ++i)
          {
          const double phi_i_r = fe_values[r].value (i, q_point);
          const double phi_i_s = fe_values[s].value (i, q_point);

          cell_rhs(i) -=  (surf_parameters->gamma)*(  (common_parameters->alpha_1)*r_loc[q_point]
                            - (common_parameters->beta_1)*bulk_loc[q_point][0] // u_loc[q_point]
                            - (common_parameters->kappa_1)*bulk_loc[q_point][1] // v_loc[q_point]
                           )*phi_i_r*JxW ;
          cell_rhs(i) -=  (surf_parameters->gamma)*(  (common_parameters->alpha_2)*s_loc[q_point]
                            - (common_parameters->beta_2)*bulk_loc[q_point][0] // u_loc[q_point]
                            - (common_parameters->kappa_2)*bulk_loc[q_point][1] //v_loc[q_point]
                           )*phi_i_s*JxW ;

        } //end for dof i
      }  // end for q_point
	  
      // add local contributions to global system
      cell->get_dof_indices (local_dof_indices);

      // get assembler lock since writing to global object
      assembler_lock.acquire ();
      for (unsigned int i=0; i<dofs_per_cell; ++i)
	 coupling_term_surf(local_dof_indices[i]) += cell_rhs(i);
     // finished writing to global object, so release assembler lock
     assembler_lock.release ();
   } //end for cell
} 



  ////////////////////
 //  Solver system //
////////////////////

// #solve_system# //

template <int dim, int spacedim>
void CoupledSystemHandler<dim,spacedim>::solve_system (BlockVector<double> &return_vector) 
{
  // use this when solving non symmetric Newton system
  SolverControl                      solver_control (100000,common_parameters->matrix_solver_tol);

//GrowingVectorMemory<BlockVector<double> > vector_memory;
//SolverGMRES<BlockVector<double> >::AdditionalData gmres_data;
//gmres_data.max_n_tmp_vectors = 100;

  //SolverGMRES<BlockVector<double> >  solver(solver_control, vector_memory,gmres_data);

  SolverBicgstab<BlockVector<double> >::AdditionalData Bicgstab_params;
    // Bicgstab_params.exact_residual = false;
    // Bicgstab_params.breakdown      = 1.e-7;
  SolverBicgstab<BlockVector<double> > solver(solver_control,Bicgstab_params);
 
 //PreconditionJacobi<BlockSparseMatrix<double> > preconditioner;
 //preconditioner.initialize(system_matrix,.9);

  //SparseILU<double> preconditioner;
  //preconditioner.initialize(system_matrix SparseILU<double>::AdditionalData());


  PreconditionIdentity preconditioner;

  solver.solve (system_matrix, return_vector, system_rhs,
               preconditioner);
               // PreconditionIdentity());
}


void check_matrix_zero_entries(BlockSparseMatrix<double> &M)
{
	for(int i=0; i<4;++i)
	  for(int j=0;j<4;++j)
   	    std::cout << "Block (" << i << "," << j << "): " 
	              << M.block(i,j).n_actually_nonzero_elements() 
	              << " of "
	              << M.block(i,j).n_nonzero_elements()
	              << " non zero elements."
	              << std::endl;
}	              

   //////////////////////
  //  BE timestep  //
 //////////////////////

template<int dim,int spacedim>
void CoupledSystemHandler<dim,spacedim>::BE_timestep()
{
 {  // initialise bulk and surf base matrices and lhs
     Threads::TaskGroup<void> task_group;
        task_group += Threads::new_task(&RDProblem<spacedim,spacedim>::BE_initialise,this->RD_bulk);
        task_group += Threads::new_task(&RDProblem<dim,spacedim>::BE_initialise,this->RD_surf); 
       task_group.join_all();
 }   	
     
  for( unsigned int nonlin_count = 1 ; nonlin_count <= (common_parameters->max_nonlin_iter); ++nonlin_count)
   {  
     lin_solution = solution;
     system_rhs = 0;
     //system_matrix = 0;
     system_matrix.copy_from(base_lhs);

  	{  // get individual system terms 
     Threads::TaskGroup<void> task_group;
        task_group += Threads::new_task(&RDProblem<spacedim,spacedim>::BE_assemble_newton,this->RD_bulk);
        task_group += Threads::new_task(&RDProblem<dim,spacedim>::BE_assemble_newton,this->RD_surf); 
       task_group.join_all();
     }   
     { // get coupling terms 
     Threads::TaskGroup<void> task_group;		 
        task_group += Threads::new_task(&CoupledSystemHandler::get_bulk_coupling_term,*this);
        task_group += Threads::new_task(&CoupledSystemHandler::get_surf_coupling_term,*this);
       task_group.join_all();
     } 
     { // add individual to full system
     Threads::TaskGroup<void> task_group;		 
        task_group += Threads::new_task(&CoupledSystemHandler::add_to_matrix,*this,
                                         1.,RD_bulk.system_matrix,0,0,system_matrix,0,0);
        task_group += Threads::new_task(&CoupledSystemHandler::add_to_matrix,*this,
                                         1.,RD_bulk.system_matrix,0,1,system_matrix,0,1);
        task_group += Threads::new_task(&CoupledSystemHandler::add_to_matrix,*this,
                                         1.,RD_bulk.system_matrix,1,0,system_matrix,1,0);       
        task_group += Threads::new_task(&CoupledSystemHandler::add_to_matrix,*this,
                                         1.,RD_bulk.system_matrix,1,1,system_matrix,1,1);
        
        task_group += Threads::new_task(&CoupledSystemHandler::add_to_matrix,*this,
                                         1.,RD_surf.system_matrix,0,0,system_matrix,2,2);
        task_group += Threads::new_task(&CoupledSystemHandler::add_to_matrix,*this,
                                         1.,RD_surf.system_matrix,0,1,system_matrix,2,3);
        task_group += Threads::new_task(&CoupledSystemHandler::add_to_matrix,*this,
                                         1.,RD_surf.system_matrix,1,0,system_matrix,3,2);
        task_group += Threads::new_task(&CoupledSystemHandler::add_to_matrix,*this,
                                         1.,RD_surf.system_matrix,1,1,system_matrix,3,3); 
       task_group.join_all();
     }  
    // add system rhs terms
  system_rhs.block(0) += RD_bulk.system_rhs.block(0);
   system_rhs.block(1) += RD_bulk.system_rhs.block(1);
  system_rhs.block(2) += RD_surf.system_rhs.block(0);
   system_rhs.block(3) += RD_surf.system_rhs.block(1);   
   // add coupling terms
     system_rhs.block(0) += coupling_term_bulk.block(0);
   system_rhs.block(1) += coupling_term_bulk.block(1);
  system_rhs.block(2) += coupling_term_surf.block(0);
   system_rhs.block(3) += coupling_term_surf.block(1); 

  solve_system(solution);
  solution += lin_solution;
    // update solution
  RD_bulk.solution.block(0) = solution.block(0);
   RD_bulk.solution.block(1) = solution.block(1);
  RD_surf.solution.block(0) = solution.block(2);
   RD_surf.solution.block(1) = solution.block(3);
   
   // now check for convergence
  RD_bulk.compute_L2norm_of_difference(RD_bulk.solution,RD_bulk.lin_solution,RD_bulk.nonlin_L2diff_xi);
  RD_surf.compute_L2norm_of_difference(RD_surf.solution,RD_surf.lin_solution,RD_surf.nonlin_L2diff_xi);
  double nonlin_L2diff = RD_bulk.nonlin_L2diff_xi + RD_surf.nonlin_L2diff_xi;
 
  /*double nonlin_L2diff = RD_bulk.nonlin_L2diff_xi*RD_bulk.nonlin_L2diff_xi 
                             + RD_surf.nonlin_L2diff_xi*RD_surf.nonlin_L2diff_xi;
  nonlin_L2diff = std::pow(nonlin_L2diff, .5);*/       
  
  /*double nonlin_L2diff;
  compute_L2norm_of_difference(lin_solution,solution,nonlin_L2diff);*/
                      
  
  shared_out << std::setprecision(5) << std::fixed << "t: " << t 
		        << " iter: " << nonlin_count;
  shared_out << std::setprecision(3) << std::scientific         
                << "  L2 diff : " << nonlin_L2diff  << std::endl;
  if (nonlin_L2diff < (common_parameters->nonlin_solver_tol) )
    break;  
   } //end newton loop
}



   //////////////////////
  //  FST timestep 1  //
 //////////////////////

template<int dim,int spacedim>
void CoupledSystemHandler<dim,spacedim>::FST_timestep_1()
{
  system_rhs = 0;
  //system_matrix = 0;
  system_matrix.copy_from(base_lhs);
 
  	{  // get system terms and coupling terms and get bulk and surf contributions
     Threads::TaskGroup<void> task_group;
        task_group += Threads::new_task(&RDProblem<spacedim,spacedim>::FST_timestep_1_assemble,this->RD_bulk);
        task_group += Threads::new_task(&RDProblem<dim,spacedim>::FST_timestep_1_assemble,this->RD_surf); 
       task_group.join_all();
     }   	
    /* { // get coupling terms 
     Threads::TaskGroup<void> task_group;		 
        task_group += Threads::new_task(&CoupledSystemHandler::get_bulk_coupling_term,*this);
        task_group += Threads::new_task(&CoupledSystemHandler::get_surf_coupling_term,*this);
       task_group.join_all();
     } */ 
     { // add individual to full system
     Threads::TaskGroup<void> task_group;		 
        task_group += Threads::new_task(&CoupledSystemHandler::add_to_matrix,*this,
                                         1.,RD_bulk.system_matrix,0,0,system_matrix,0,0);
        task_group += Threads::new_task(&CoupledSystemHandler::add_to_matrix,*this,
                                         1.,RD_bulk.system_matrix,0,1,system_matrix,0,1);
        task_group += Threads::new_task(&CoupledSystemHandler::add_to_matrix,*this,
                                         1.,RD_bulk.system_matrix,1,0,system_matrix,1,0);       
        task_group += Threads::new_task(&CoupledSystemHandler::add_to_matrix,*this,
                                         1.,RD_bulk.system_matrix,1,1,system_matrix,1,1);
        
        task_group += Threads::new_task(&CoupledSystemHandler::add_to_matrix,*this,
                                         1.,RD_surf.system_matrix,0,0,system_matrix,2,2);
        task_group += Threads::new_task(&CoupledSystemHandler::add_to_matrix,*this,
                                         1.,RD_surf.system_matrix,0,1,system_matrix,2,3);
        task_group += Threads::new_task(&CoupledSystemHandler::add_to_matrix,*this,
                                         1.,RD_surf.system_matrix,1,0,system_matrix,3,2);
        task_group += Threads::new_task(&CoupledSystemHandler::add_to_matrix,*this,
                                         1.,RD_surf.system_matrix,1,1,system_matrix,3,3); 
       task_group.join_all();
     }  
    // add system rhs terms
  system_rhs.block(0) += RD_bulk.system_rhs.block(0);
   system_rhs.block(1) += RD_bulk.system_rhs.block(1);
  system_rhs.block(2) += RD_surf.system_rhs.block(0);
   system_rhs.block(3) += RD_surf.system_rhs.block(1);   
   // add coupling terms
  /*   system_rhs.block(0) += coupling_term_bulk.block(0);
   system_rhs.block(1) += coupling_term_bulk.block(1);
  system_rhs.block(2) += coupling_term_surf.block(0);
   system_rhs.block(3) += coupling_term_surf.block(1); */

  solve_system(solution);
    // update solution
  RD_bulk.solution.block(0) = solution.block(0);
   RD_bulk.solution.block(1) = solution.block(1);
  RD_surf.solution.block(0) = solution.block(2);
   RD_surf.solution.block(1) = solution.block(3);
}

   //////////////////////
  //  FST timestep 2  //
 //////////////////////

template<int dim,int spacedim>
void CoupledSystemHandler<dim,spacedim>::FST_timestep_2()
{
  	{  // get system terms and coupling terms and get bulk and surf contributions
     Threads::TaskGroup<void> task_group;
        task_group += Threads::new_task(&RDProblem<spacedim,spacedim>::FST_timestep_2_initialise,this->RD_bulk);
        task_group += Threads::new_task(&RDProblem<dim,spacedim>::FST_timestep_2_initialise,this->RD_surf); 
       task_group.join_all();
    }		
    { // get coupling terms 
     Threads::TaskGroup<void> task_group;		 
        task_group += Threads::new_task(&CoupledSystemHandler::get_bulk_coupling_term,*this);
        task_group += Threads::new_task(&CoupledSystemHandler::get_surf_coupling_term,*this);
       task_group.join_all();
    }  
  
  for( unsigned int nonlin_count = 1 ; nonlin_count <= (common_parameters->max_nonlin_iter); ++nonlin_count)
   {  
     lin_solution = solution;
     //system_matrix = 0;
     //system_matrix.copy_from(base_lhs);
     //system_rhs = 0;
       	 
  	{  // get individual system terms 
     Threads::TaskGroup<void> task_group;
        task_group += Threads::new_task(&RDProblem<spacedim,spacedim>::FST_timestep_2_assemble_newton,this->RD_bulk);
        task_group += Threads::new_task(&RDProblem<dim,spacedim>::FST_timestep_2_assemble_newton,this->RD_surf); 
       task_group.join_all();
    }   
  // solve within the classes themselves since it's quicker   
       // add coupling terms
  RD_bulk.system_rhs += coupling_term_bulk;
  RD_surf.system_rhs += coupling_term_surf;

     //solve
  RD_bulk.solve_block_system_whole(RD_bulk.solution);
  RD_surf.solve_block_system_whole(RD_surf.solution);
     
     RD_bulk.solution += RD_bulk.lin_solution;
     RD_surf.solution += RD_surf.lin_solution;
     
   solution.block(0) = RD_bulk.solution.block(0);
   solution.block(1) = RD_bulk.solution.block(1);
   solution.block(2) = RD_surf.solution.block(0);
   solution.block(3) = RD_surf.solution.block(1);
          
     /*{ // add individual to full system
     Threads::TaskGroup<void> task_group;		 
        task_group += Threads::new_task(&CoupledSystemHandler::add_to_matrix,*this,
                                         1.,RD_bulk.system_matrix,0,0,system_matrix,0,0);
        task_group += Threads::new_task(&CoupledSystemHandler::add_to_matrix,*this,
                                         1.,RD_bulk.system_matrix,0,1,system_matrix,0,1);
        task_group += Threads::new_task(&CoupledSystemHandler::add_to_matrix,*this,
                                         1.,RD_bulk.system_matrix,1,0,system_matrix,1,0);       
        task_group += Threads::new_task(&CoupledSystemHandler::add_to_matrix,*this,
                                         1.,RD_bulk.system_matrix,1,1,system_matrix,1,1);
        
        task_group += Threads::new_task(&CoupledSystemHandler::add_to_matrix,*this,
                                         1.,RD_surf.system_matrix,0,0,system_matrix,2,2);
        task_group += Threads::new_task(&CoupledSystemHandler::add_to_matrix,*this,
                                         1.,RD_surf.system_matrix,0,1,system_matrix,2,3);
        task_group += Threads::new_task(&CoupledSystemHandler::add_to_matrix,*this,
                                         1.,RD_surf.system_matrix,1,0,system_matrix,3,2);
        task_group += Threads::new_task(&CoupledSystemHandler::add_to_matrix,*this,
                                         1.,RD_surf.system_matrix,1,1,system_matrix,3,3); 
       task_group.join_all();
     }         
  // add system rhs terms
  system_rhs.block(0) += RD_bulk.system_rhs.block(0);
   system_rhs.block(1) += RD_bulk.system_rhs.block(1);
  system_rhs.block(2) += RD_surf.system_rhs.block(0);
   system_rhs.block(3) += RD_surf.system_rhs.block(1);
  
  // add coupling terms
  system_rhs.block(0) += coupling_term_bulk.block(0);
   system_rhs.block(1) += coupling_term_bulk.block(1);
  system_rhs.block(2) += coupling_term_surf.block(0);
   system_rhs.block(3) += coupling_term_surf.block(1);

  solve_system(solution);
  solution += lin_solution;

  RD_bulk.solution.block(0) = solution.block(0);
   RD_bulk.solution.block(1) = solution.block(1);
  RD_surf.solution.block(0) = solution.block(2);
   RD_surf.solution.block(1) = solution.block(3);*/
    
  RD_bulk.compute_L2norm_of_difference(RD_bulk.solution,RD_bulk.lin_solution,RD_bulk.nonlin_L2diff_xi);
  RD_surf.compute_L2norm_of_difference(RD_surf.solution,RD_surf.lin_solution,RD_surf.nonlin_L2diff_xi);
  double nonlin_L2diff = RD_bulk.nonlin_L2diff_xi + RD_surf.nonlin_L2diff_xi;
 
  /*double nonlin_L2diff = RD_bulk.nonlin_L2diff_xi*RD_bulk.nonlin_L2diff_xi 
                             + RD_surf.nonlin_L2diff_xi*RD_surf.nonlin_L2diff_xi;
  nonlin_L2diff = std::pow(nonlin_L2diff, .5);*/       
  
  /*double nonlin_L2diff;
  compute_L2norm_of_difference(lin_solution,solution,nonlin_L2diff);*/
                      
  
  shared_out << std::setprecision(5) << std::fixed << "t: " << t 
		        << " iter: " << nonlin_count;
  shared_out << std::setprecision(3) << std::scientific         
                << "  L2 diff : " << nonlin_L2diff  << std::endl;
  if (nonlin_L2diff < (common_parameters->nonlin_solver_tol) )
    break;  
   } //end newton loop
}


template <int dim, int spacedim>
void CoupledSystemHandler<dim,spacedim>::compute_L2norm_of_difference (const BlockVector<double> &solution_1,
                                                                   const BlockVector<double> &solution_2,
                                                                   double                    &return_value)
{
// return vector must have two elements
BlockVector<double>  tmp_diff = solution_1;
tmp_diff -= solution_2;

return_value =tmp_diff.l2_norm();

}


   ///////////
  //  Run  //
 ///////////

template<int dim,int spacedim>
void CoupledSystemHandler<dim,spacedim>::run()
{
  initialise();
  shared_out << "Starting time-loop..." << std::endl;
    // time loop
   switch(common_parameters->timestep_method){
     case FST:
        for( t=common_parameters->tau;t <= common_parameters -> T; t += common_parameters -> tau){  
		  old_solution = solution;
          RD_bulk.update_time();
          RD_surf.update_time();
          			
	      FST_timestep_1();
	      FST_timestep_2();
          FST_timestep_1();

          RD_bulk.process_solution();
          RD_surf.process_solution();
          
         dotted_line();
         double solution_L2diff;
             compute_L2norm_of_difference(old_solution,solution,solution_L2diff);
         if(solution_L2diff<1.e-10 )
              break; }
         break;
       case BE:
        for(t=common_parameters->tau;t <= common_parameters -> T; t += common_parameters -> tau){
  		  old_solution = solution;
          RD_bulk.update_time();
          RD_surf.update_time();
          			
	      BE_timestep();

          RD_bulk.process_solution();
          RD_surf.process_solution();
          
         dotted_line();
         double solution_L2diff;
             compute_L2norm_of_difference(old_solution,solution,solution_L2diff);
         if(solution_L2diff<1.e-10 )
              break; }      
         break;
       case CN:
        for(double t=common_parameters->tau;t <= common_parameters -> T; t += common_parameters -> tau){
            shared_out << " Bulk:" << std::endl;
           get_bulk_coupling_term();
           RD_bulk.CN_timestep(coupling_term_bulk);
            shared_out << " Surface:" << std::endl;
           get_surf_coupling_term();
           RD_surf.CN_timestep(coupling_term_surf); 
         dotted_line();
           if( ( RD_bulk.solution_L2diff[0]<.01*(bulk_parameters->matrix_solver_tol))
              &&( RD_surf.solution_L2diff[1]<.01*(surf_parameters->matrix_solver_tol)) )
                break;}
         break;
     } //end switch

     RD_surf.output_solution(RD_surf.solution,surf_output_directory + "/xi_final.vtu");
     RD_bulk.output_solution(RD_bulk.solution,bulk_output_directory + "/xi_final.vtu");


}

 ///////////////////
// Main function //
//////////////////
 
int main (int argc, char **argv) 
{
  // note time at beginning
  std::clock_t t1,t2;
  t1=std::clock();
  
  // surpress some console output
  deallog.depth_console (0);

  // initialise random number generator with time 
  //std::srand((unsigned)std::time(0)); 

  // get input parameter file
  std::string common_parameter_filename,
              bulk_parameter_filename,
              surf_parameter_filename;
  if (argc < 2)
    {
     common_parameter_filename = "schnak.param";
     bulk_parameter_filename = "schnak.param_bulk";
     surf_parameter_filename = "schnak.param_surf";
    }
  else
    {
     common_parameter_filename = argv[1];
     bulk_parameter_filename = argv[2];
     surf_parameter_filename = argv[3];
    }

   
     CommonParameters common_parameters(common_parameter_filename);
     Parameters       bulk_parameters(bulk_parameter_filename),
                      surf_parameters(surf_parameter_filename);   

     common_parameters.declare_parameters();
     common_parameters.get_parameters();
     common_parameters.print_parameters();
     bulk_parameters.declare_parameters();
     bulk_parameters.get_parameters();
     surf_parameters.declare_parameters();
     surf_parameters.get_parameters();
 
    /*CoupledSystemHandler<2,3>* coupledsystemhandler ;
    coupledsystemhandler = new() CoupledSystemHandler<2,3>(common_parameters,
                                                   bulk_parameters,
                                                   surf_parameters);
    coupledsystemhandler -> run();
    delete coupledsystemhandler;*/

    CoupledSystemHandler<2,3> coupledsystemhandler(common_parameters,
                                                   bulk_parameters,
                                                   surf_parameters);
    coupledsystemhandler.run();

  
 
 // note time at end
 t2 = std::clock();
 
 // work out time spent in program and output on terminal
 float diff ((float)t2-(float)t1);
 std::cout << "Success: Execution time, " 
           << diff /CLOCKS_PER_SEC << " secs" << std::endl
           << "======================"    << std::endl  ;  ;


 return 0;
}

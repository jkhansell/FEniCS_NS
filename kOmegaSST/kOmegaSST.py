from collections import namedtuple
import gmsh
import numpy as np 
from mpi4py import MPI
from dolfinx.io import gmshio

from petsc4py import PETSc 
from dolfinx import mesh, fem, io, nls, plot 
import ufl

import pyvista as pv

class kOmegaSST():

    def __init__(self, mesh_file, gdim, num_viscosity, wall_dist_tol, 
                    wall_dist_iters):

        self.mesh_file = mesh_file          # gmsh model 
        self.mesh_comm = MPI.COMM_WORLD     # 
        self.model_rank = 0
        self.gdim = gdim
        self.h = num_viscosity
        self.epseik = wall_dist_tol
        self.eikiters = wall_dist_iters
        self.load_domain("airfoil")
        self.model_constants()


    def model_constants(self):
        self.a_1 = 0.31
        self.betastar = 0.09
        self.sigmak = np.array([0.85, 1.])
        self.sigmaomega = np.array([0.5, 0.856])
        self.beta = np.array([0.075, 0.0828])
        self.kappa = 0.41
        self.gamma = self.beta/self.betastar - \
                self.sigmaomega*self.kappa**2/np.sqrt(self.betastar)

      
    def load_domain(self, name):
        gmsh.initialize()
        gmsh.model.add(name)
        gmsh.open(self.mesh_file)
        self.domain, _, ft = gmshio.model_to_mesh(gmsh.model, 
                                self.mesh_comm, self.model_rank, gdim=self.gdim)
        gmsh.finalize()


    def get_wall_distance(self):
        """
            This function solves for the eikonal equation |grad(phi)|^2 = 1
            using a viscous approach eps*lap(phi) + |grad(phi)|^2 = 1 as a 
            first guess, then several iterations on solving for phi = phi_0 + dphi 
        """


        # FEniCSx is used to solve for the specific geometry
        # Set function space 
        V = fem.FunctionSpace(self.domain, ("CG", 1))
        
        # Set boundary functions
        phi_boundary = fem.Function(V)

        print(type(phi_boundary))
        phi_boundary.interpolate(lambda x: 0.0 *(x[0]+x[1]))

        boundary_dim = self.gdim - 1

        boundary_facets = mesh.locate_entities_boundary(self.domain, boundary_dim,
            lambda x: np.full(x.shape[1], True, dtype=bool))

        bc = fem.dirichletbc(phi_boundary, fem.locate_dofs_topological(V, boundary_dim,
            boundary_facets))
        
        # Set up variational formulation

        phi0 = fem.Function(V)
        v = ufl.TestFunction(V)
        phi0.interpolate(lambda x: 0.0*(x[0]+x[1]))

        # Set up residual function 
        Residual = (self.h*ufl.inner(ufl.grad(phi0), ufl.grad(v)) \
            + ufl.inner(ufl.grad(phi0), ufl.grad(phi0))*v-v)*ufl.dx
        
        # solve problem via Newton method

        problem = fem.petsc.NonlinearProblem(Residual, phi0, bcs=[bc])
        solver = nls.petsc.NewtonSolver(MPI.COMM_WORLD, problem)
        solver.convergence_criterion = "incremental"
        solver.rtol = 1e-1
        solver.report = True

        n, converged = solver.solve(phi0)

        assert(converged)
        
        # solve dphi perturbation 
        
        err = 1
        i = 0 
        while err > self.epseik and i < self.eikiters:
            v = ufl.TestFunction(V)
            dphi = ufl.TrialFunction(V)

            A = (self.h*ufl.inner(ufl.grad(dphi), ufl.grad(v))
                +2*(ufl.inner(ufl.grad(phi0), ufl.grad(dphi))*v))*ufl.dx

            L = (v - self.h*ufl.inner(ufl.grad(phi0),ufl.grad(v)) 
                - ufl.inner(ufl.grad(phi0)**2, v))*ufl.dx

            problem = fem.petsc.LinearProblem(A, L, bcs=[bc], \
                                petsc_options={"ksp_type": "gmres",
                                               "pc_type": "lu"})
            dphi = problem.solve()

            phi0.x.array.real += dphi.x.array.real

            grad = fem.assemble_scalar(fem.form(ufl.inner(dphi,dphi)*ufl.dx))
            err = (self.domain.comm.allreduce(grad,op=MPI.SUM))
            i += 1    

        self.phi = (phi0, V)

    def plot_wall_distance(self): 
        """
            Plot the wall distance.
        """

        phi_topology, phi_cell_types, phi_geometry = plot.create_vtk_mesh(self.phi[1])
        phi_grid = pv.UnstructuredGrid(phi_topology, phi_cell_types, phi_geometry)
        phi_grid.point_data["phi"] = self.phi[0].x.array.real
        phi_grid.set_active_scalars("phi")
        phi_plotter = pv.Plotter()
        phi_plotter.add_mesh(phi_grid, show_edges=False)
        phi_plotter.view_xy()
        if not pv.OFF_SCREEN:
            phi_plotter.show()

    def setup_Uproj(self, V):
        """
            Set up the U-projection.
        """

        U = ufl.TrialFunction(V) 
        v = ufl.TestFunction(V)

        

         



"""
    def set_initial_conditions(self):
        i = 0 

    def sigma2(U):

        return 2*ufl.sym(ufl.grad(U))


    def F2(self, k, omega, y, nu):

        arg1 = (2/0.09)*(ufl.sqrt(k)/(omega*y)) 
        arg2 = 500*nu/(omega*y**2)
        arg = ufl.max_value(arg1, arg2)**2

        return ufl.tanh(arg)

    def mu_t(self, U, k, omega, rho, y, nu): 

        magvort = ufl.sqrt(ufl.inner(ufl.curl(U)))
        arg = ufl.max_value(self.a_1*omega, magvort*self.F2(k, omega, y, nu))

        return rho*self.a_1*k/arg


    def tau(self, U, k):
        return self.mu_t()*self.sigma2(U) - (2/3)*self.rho*k

    def set_up_U_eqn(self, V, dt, mu):
        U = ufl.TrialFunction(V) 
        v = ufl.TestFunction(V)
        
        # TEST FUNCTION DIVISION, AND MAKE EVERYTHING "SELF"

        lhs = ((U-self.U_1)/dt*v*ufl.dx + (ufl.inner(U, ufl.grad(U)))*v*ufl.dx
                - ufl.inner((mu+self.mu_t())*self.sigma2 - (2/3)*self.rho*ufl.grad(self.k)), ufl.grad(v))



        # forward euler scheme

    def Unsteady_Solver(self):
        V = fem.FunctionSpace(self.domain, ("CG", 1))
        
    """

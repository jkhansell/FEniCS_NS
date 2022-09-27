import gmsh
import numpy as np

gmsh.initialize()
gmsh.model.add("airfoil")

L = 2
gdim = 2 
airfoildata = np.loadtxt("airfoil.txt", delimiter=",")
AFcentroid = np.mean(airfoildata, axis=0)

tags = []
for i in range(len(airfoildata)):
    gmsh.model.occ.addPoint(airfoildata[i,0], airfoildata[i,1], 0.0, tag=i)
    tags.append(i)

tags.append(0)

spl = gmsh.model.occ.addSpline(tags, 1)
airfoilloop = gmsh.model.occ.addCurveLoop([spl])
gmsh.model.occ.mesh.setSize(gmsh.model.occ.getEntities(0), 0.008)

p1 = gmsh.model.occ.addPoint(AFcentroid[0]-1.5*L, AFcentroid[1]-L,0.0)
p2 = gmsh.model.occ.addPoint(AFcentroid[0]+1.5*L, AFcentroid[1]-L,0.0)
p3 = gmsh.model.occ.addPoint(AFcentroid[0]+1.5*L, AFcentroid[1]+L,0.0)
p4 = gmsh.model.occ.addPoint(AFcentroid[0]-1.5*L, AFcentroid[1]+L,0.0)

l1 = gmsh.model.occ.addLine(p1, p2)
l2 = gmsh.model.occ.addLine(p2, p3) 
l3 = gmsh.model.occ.addLine(p3, p4)
l4 = gmsh.model.occ.addLine(p4, p1)

cout = gmsh.model.occ.addCurveLoop([l1, l2, l3, l4])
airfoil = gmsh.model.occ.addPlaneSurface([cout, airfoilloop])

gmsh.model.occ.synchronize()

gmsh.model.addPhysicalGroup(2, [airfoil], name="Fluid")
gmsh.model.addPhysicalGroup(1, [l4] , name="inflow")
gmsh.model.addPhysicalGroup(1, [l2] , name="outflow")
gmsh.model.addPhysicalGroup(1, [l1,l3] , name="walls")

gmsh.model.occ.synchronize() 

gmsh.model.mesh.field.add("Distance", 1)
gmsh.model.mesh.field.setNumbers(1, "CurvesList", [spl])
gmsh.model.mesh.field.setNumber(1, "Sampling", 100)

gmsh.model.mesh.field.add("Threshold", 2)
gmsh.model.mesh.field.setNumber(2, "InField", 1)
gmsh.model.mesh.field.setNumber(2, "SizeMin", 0.02)
gmsh.model.mesh.field.setNumber(2, "SizeMax", 0.2)
gmsh.model.mesh.field.setNumber(2, "DistMin", 0.1)
gmsh.model.mesh.field.setNumber(2, "DistMax", 0.5)

gmsh.model.mesh.field.setAsBackgroundMesh(2)

gmsh.model.mesh.generate(2)

gmsh.write("airfoil.msh")
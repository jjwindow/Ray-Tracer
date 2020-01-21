# -*- coding: utf-8 -*-
"""
Created on Wed Feb 20 16:06:08 2019

@author: JJW3517

Module containing classes used in the 2nd year physics ray tracer computing 
project.

Ray                 - Class for light ray in 3d space with a list of points and 
                      directions.
OpticalElement      - Base class for any optical element.
SphericalRefraction - Derived from OpticalElement, represents a spherical 
                      optical element surface. Allows a ray to be propagated
                      through and performs refraction.
OutputPlane         - Derived from OpticalElement, represents the plane at which
                      the rays are stopped.
RayBundle           - Multiple instances of Ray with starting points that form
                      a circle of constant point density.

"""
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.optimize import fmin_tnc

csfont = {'family':'serif'}
fixed_thickness = 5
fixed_plane = 200
fixed_curvs = [0.05, -0.05]

class Ray:
    """
    Class representing a ray of light. Instantiated with a starting point and
    a direction vector.
    ATTRIBUTES: 
        -point = [0,0,0], the starting position of the ray as a vector.
        -direction = [0,0,1], the direction of the ray as a vector.
    """
        
    def __init__(self, point=[0,0,0], direction=[0,0,1]):
        self._point = np.array(point)
        norm = np.linalg.norm(direction)
        self._direction = np.array(direction)/norm # normalises direction vector
        
        self._ray_points = []
        self._ray_k = []
        
        self._ray_points.append(np.array(point))
        self._ray_k.append(np.array(direction))
        
    def __repr__(self):
        """
        String representation.
        """
        return "{}({},{})".format("Ray", self.p(), self.k())
        
    def p(self):
        """
        Returns the current point vector of the ray as a Numpy array
        """
        return self._point
    
    def k(self):
        """
        Returns the current direction vector of the ray as a Numpy array
        """
        return self._direction
    
    def append(self, p, k):
        """
        Method for appending values of position p and direction k to the ray 
        object.
        """
        p = np.array(p)
        k = np.array(k)
        self._point = p
        self._direction = k
        self._ray_points.append(p)
        self._ray_k.append(k)
        return self
        
    def vertices(self):
        """
        Returns the list of the ray's vertices.
        """
        return self._ray_points
    
    def terminate(self):
        """
        Terminates the ray by inserting zeros in the lists of vertices and 
        directions.
        """
        self._ray_points.append(0)
        self._ray_k.append(0)
        return self
    
    def _unpropagate(self):
        """
        Deletes the most recent vertex and direction added to the lists. Used 
        in the optimisation to remove propagations through temporary output
        planes.
        """
        del self._ray_points[-1]
        del self._ray_k[-1]
        return self
    
class OpticalElement:
    """
    Base class of optical elements SphericalRefraction and OutputPlane.
    """
    def propagate_ray(self, ray):
        "propagate a ray through the optical element"
        raise NotImplementedError()
    
class SphericalRefraction(OpticalElement):
    """
    Spherical refraction element derived from the optical element class. 
    ATTRIBUTES:
        z0 - axis intercept
        n1 - refractive index of material before boundary
        n2 - refractive index of material after boundary (element material)
        curve - curvature of the spherical element
        r_aper - perpendicular radius which subtends the element from the sphere
        
    """
    def __init__(self, z0, n1, n2, curve, r_aper):
        self._intercept = float(z0)
        self.n1 = n1
        self.n2 = n2
        self._curvature = curve
        self._aperture = r_aper
        if self._curvature < 0:
            self._lens_endpoint = z0 - abs(1/self._curvature)
        else:
            self._lens_endpoint = self._intercept
        
        return None
        
    def intercept(self, ray):
        """
        Calculates the first valid intercept between the ray and
        the optical element.
        """
        P = ray.p()
        k = ray.k()
        if self._curvature != 0:

            R = 1/self._curvature
            O = np.array([0,0, self._intercept]) + np.array([0,0,R])
            self._centre = O
            r = P-O
        
            #calculate quadratic for l
            b = np.dot(r, k)
            r_sq = np.dot(r, r)
            discr = b**2 - (r_sq - R**2)
            if discr < 0:
                return None #discard imaginary solutions
            else:
                l_plus = -b + np.sqrt(discr)
                l_min = -b - np.sqrt(discr)
                
                #Selection criteria 1: remove negative magnitudes
                if (l_plus<0 and l_min<0):
                    return None
                elif l_min<0 and l_plus >= 0:
                    l = l_plus
    
                #Selection criteria 2: select correct solution in case of
                #two positive solutions
                else:
                    if self._curvature < 0:
                        l = max(l_min, l_plus) # selects longer l if curvature is negative
                    elif self._curvature > 0:
                        l = min(l_min, l_plus) # selects shorter l for positive curvature. 
                
                Q = P + l*k # finds intersection point Q (vector).
                
                #Selection criteria 3: Check intersection is inside sphere.
                ray_axis_dist = np.sqrt(Q[0]**2 + Q[1]**2) 
                if ray_axis_dist > self._aperture:
                    return None
                
                #Also check that intersection is not past the lens element
                if self._curvature > 0:
                    if Q[2] < self._lens_endpoint:
                        check = round(Q[2], 3) #prevents this clause being 
                        #called for rounding errors, e.g - 99.9999<100.
                        if check == self._lens_endpoint:
                            return Q
                        return None                
                elif self._curvature < 0:
                    if Q[2] < self._lens_endpoint:
                        return None
                return Q
        else:
            #dealing with a planar element case
            z = P[2]
            z0 = self._intercept
            self._lens_endpoint = z0 #will be used for output plane
            k_z = k[2]
            
            if k_z != 0:
                l = (z0-z)/k_z
            else:
                #No valid intercept, ray is parallel to planar element"
                return None
            if l<0:
                #Ray starts past the element, no intercept.
                return None
            else:
                Q = P + l*k #intersection vector Q
                Q_r = np.sqrt(Q[0]**2+Q[1]**2)
                if Q_r > self._aperture:
                    #No valid intercept found, intersection point is outside of
                    #element aperture
                    return None
                else:
                    return Q
    
    def snell(self, k_in, n):
        """
        Gives the direction of a ray refracted through a surface
        with a given normal vector. Uses the equation for a refracted
        direction vector using Snell's law.
        
        Returns a unit vector in the direction of the refracted ray.
        
        If the condition for total internal reflection is met, then None
        returned.
        
        ***NOTE: Unit normal vector is defined in the -z direction
                 (away from the surface of the element).***
        """
        k_in = k_in/np.linalg.norm(k_in)
        n = n/np.linalg.norm(n)
        
        r = self.n1 / self.n2  
        c = -np.dot(n, k_in) #cosine of incident angle
        sin_in = np.sqrt(1-c**2)

        if sin_in > 1/r:
            print("Total Internal Reflection")
            return None
        else:
            #apply formula
            k_out = r*k_in + (r*c - np.sqrt(1-r**2*(1-c**2)))*n
            if k_out[2]<0:
                return None 
            return k_out
    
    def propagate_ray(self, ray):
        """
        Method to propagate a ray with given direction and origin
        through an optical element.
        
        -Finds intercept point between ray and element
        -Finds direction of refraction after element
        -Appends interception point and new direction to ray.
        
        If no valid intercept or refraction direction is found then an 
        exception is raised and the ray is terminated.
        """

        Q = self.intercept(ray)
        if type(Q) is not np.ndarray:
            ray.terminate()
            return self
            #raise Exception("No valid intercept found")
        if self._curvature > 0:
            normal_not_unit = Q-self._centre
        elif self._curvature < 0:
            normal_not_unit = self._centre-Q
        else:
            normal_not_unit = np.array([0,0,-1])
            
        n_mag = np.linalg.norm(normal_not_unit)
        n = normal_not_unit/n_mag
        
        k_in = ray.k()
        print(k_in)
        k_out = self.snell(k_in, n)
        print(k_out)
        
        if type(k_out) is not np.ndarray:
            ray.terminate()
            #raise Exception("No valid refracted ray found")
        else:
            ray.append(Q, k_out)
        return self
    
class OutputPlane(OpticalElement):
    """
    The optical output plane, centred at z = 'dist' (type: int or float). 
    Propagates the ray but does not perform refraction. Functions as a lens
    of 0 curvature with n1=n2.
    
    ATTRIBUTES: 
        -position z0
    METHODS:
        -intercept(ray)
        -propagate_ray(ray)
    """
    def __init__(self, dist):
        """
        Takes argument dist - the z-coordinate of the output plane (type: int).
        """
        self._z = dist 
        return None
    
    def intercept(self, ray):
        """
        Method to find the intercept between the propagating ray and the plane.
        Returns a position vector as a numpy.array().
        """
        z_q = ray.p()[2] #z coordinate of the last point on the ray
        k_z = ray.k()[2]

        if self._z is not None:
            l = (self._z - z_q)/k_z #will always be positive due to defn of self._z
        
            output_point = ray.p() + l*ray.k()
            return output_point
    
    def propagate_ray(self, ray):
        """
        Method to propagate the ray 'ray' to the optical plane. Uses intercept()
        to find the meeting point between the refracted ray and the plane and 
        appends this to ray.vertices()
        """
        output_point = self.intercept(ray)
        
        #there should be no instance where None is returned from intercept() as
        #the rays should always meet the plane somewhere, so no exceptions need 
        #be raised.
        
        if output_point is not None:
            ray.append(output_point, ray.k()) #direction unchanged
            return self
        
class RayBundle:
    """
    A 'bundle' of instances of Ray() with starting points that form a circle of
    constant ray density. Centred at 'centre' (type: numpy.ndarray()), 
    organised in 'ring_number' concentric rings (type: int or whole number 
    float) with a bundle radius of 'bundle_radius' (type: int or float). 
    Generates a list of starting vectors which are used to instantiate rays.
    """
    def __init__(self, bundle_radius, ring_number, centre=np.array([0,0,0]), 
                 k = np.array([0,0,1])):
        """
        Initialises a bundle of rays centred at 'centre' (type: numpy.ndarray()), 
        organised in 'ring_number' concentric rings (type: int or whole number 
        float) with a bundle radius of 'bundle_radius' (type: float). Generates 
        a list of starting vectors which are used to instantiate rays.
        """
        self._max_radius = bundle_radius
        self._rings = ring_number
        self._offset = centre
        self._direction = k
        
        self._indices = np.linspace(0, self._rings, self._rings + 1) 
        scale = self._max_radius / self._rings
        self._radii = scale*self._indices
        
        n = [1]
    
        for i in range(0,len(self._indices)):
            n.append(6*(self._indices[i]+1)) #generate number of rays in ring i
        
        self._ray_p = []

        #procedurally generates the starting vectors of parallel rays.
        for i in range(0, len(n)-1):
            for j in range(0, int(n[i])):
                theta = 2*np.pi*j/n[i]
                r = (self._radii[i]*np.array([np.cos(theta), np.sin(theta), 0]) 
                + self._offset)
                self._ray_p.append(r)
                
        self._rays = []
        
        for r in self._ray_p:
            self._rays.append(Ray(r, self._direction))
            
        return None
    def __repr__(self):
        """
        String representation.
        """
        return "{}(ring_num={}, radius={))".format("ParallelRayBundle", 
                self._rings, self._max_radius)
            
    def propagate(self, element):
        """
        Propagates the parallel ray bundle through an optical element. Uses the 
        OpticalElement.propagate_ray() method. The RayBundle class was given 
        its own propagate() method as it was easier than making a separate 
        method in OpticalElement for bundle propagation.
        """
        for ray in self._rays:
            element.propagate_ray(ray)   
        return self
    
    def _unpropagate(self):
        """
        The ray bundle analogue of Ray._unpropagate(). Applies Ray._unpropagate()
        to every ray in self.
        """
        for ray in self._rays:
            ray._unpropagate()   
        return self
                
    def plot3d(self):
        """
        Propagates the parallel ray bundle with direction vector 'k' 
        (type: numpy.ndarray) through the optical elements 'lens' and 'plane'.
        Can only be used for one lens. Plots the rays in 3 dimensions.
        """
        
        fig = plt.figure()
        ax = fig.gca(projection='3d')
        
        for ray in self._rays:
            x = []
            y = []
            z = []
            for point in ray.vertices():
                if type(point) is int:
                    break
                else:
                    y.append(point[0])
                    z.append(point[1])
                    x.append(point[2])
                    
                    ax.plot(x, y, z, color = "blue")
        plt.show()
        return self
        
    def plot2d(self):
        """
        Propagates the parallel ray bundle with direction vector 'k' 
        (type: numpy.ndarray) through the optical elements 'lens' and 'plane'.
        Can only be used for one lens. Plots the rays in 2d in both planes
        parallel to 'k'.
        """
        for ray in self._rays:
            x = []
            y = []
            z = []
            fig1 = plt.figure(1)
            plt.rcParams["figure.figsize"] = [8,8]
            plt.xlabel("z-axis (mm)", fontsize = 15, **csfont)
            plt.ylabel("x-axis (mm)", fontsize = 15, **csfont)
            plt.title("2D Ray Trace: z-x Plane")
            for point in ray.vertices():
                if type(point) is int:
                #Checks if element is 0 and exits the loop. A zero would come
                #from the Ray.terminate() function.
                    break
                else:
                    x.append(point[0])
                    y.append(point[1])
                    z.append(point[2])
                    
                    plt.plot(z, x, color = "blue")

            fig2 = plt.figure(2)
            plt.rcParams["figure.figsize"] = [8,8]
            plt.xlabel("z-axis (mm)", fontsize = 15, **csfont)
            plt.ylabel("y-axis (mm)", fontsize = 15, **csfont)
            plt.title("2d Ray Trace: z-y Plane")
            for point in ray.vertices():
                if type(point) is int:
                    break
                else:
                    plt.plot(z, y, color = "red")
            
        plt.show()
        
        return self
    
    def spotdiagram(self):
        """
        Produce a spot diagram of the rays at the output plane. 
        
        ***Uncomment first section for spot diagram in the z=0 plane***
        """
        ### Z=0 PLANE SPOT DIAGRAM ###
#        print("Spot")
#        plt.figure()
#        plt.rcParams["figure.figsize"] = [8,8]
#        plt.title("z=0")
#        
#        for p in self._ray_p:
#            plt.scatter(p[0], p[1], 1, color = "blue")
#        
#        plt.show()
        
        ### OUTPUT PLANE SPOT DIAGRAM ###    
        plt.figure()
        plt.rcParams["figure.figsize"] = [6,6]
        plt.xlabel("x axis (mm)", fontsize=15, **csfont)
        plt.ylabel("y axis (mm)", fontsize=15, **csfont)
        plt.tight_layout()
        #plt.title("Focus")

        for ray in self._rays:
            plt.scatter(ray.p()[0], ray.p()[1], 1, color = "indigo")

        plt.show()
        return self
    
    def RMS(self):
        """
        Return andthe RMS spot radius of the rays at the output plane.
        """
        s_r_sq = []
        for ray in self._rays:
            if ray.p().any() == None:
                return None
            else:
                r = ray.p()[0]**2+ray.p()[1]**2
                s_r_sq.append(r) # appends axial distance squared to array

        self._s_r_RMS = np.sqrt(sum(s_r_sq)/len(s_r_sq)) #calculates RMS radial distance
#        print("RMS SPOT RADIUS: ", self._s_r_RMS)
        return self._s_r_RMS
    
    def paraxial(self, initial_guess):
        """
        Finds the paraxial focus of the lens to a precision of 0.001mm.
        Loops through z-values iteratively from z = initial_guess onwards. At
        every value, the RMS spot radius is calculated by propagating the
        bundle through a plane at z and using RayBundle.RMS(). The loop 
        continues until an iteration where the geometrical focus is greater 
        than the previous calculation of geometrical focus. Returns the z value 
        of the paraxial focus.
        
        This method assumes that there will be a paraxial focus to be found.
        In the instance of a diverging bundle or a parallel bundle then an 
        exception will be raised when z = 1000. Howevever as this means 1 
        million ray bundle propagations it is warned that it is only suitable 
        to use this method when the bundle definitely converges.
        
        NOTE: the method _unpropagate() was written and used here to counter the 
        problem of appending multiple vertices and directions to the lists 
        self._ray_vertices and self._ray_k. It does not represent a physical 
        process or property of lenses.
        """
        z = initial_guess
        step = 0.001
        cont = True
        prev = 100
        
        while cont==True:
            if z >= 1000:
                raise Exception("z=1000, too many iterations. Check setup.")
            
            plane = OutputPlane(z)
            self.propagate(plane)
            RMS = self.RMS()
            
            if RMS <= prev:
                prev = RMS
                z += step
                self._unpropagate()
            else:
                cont = False
                self._unpropagate()
                self.propagate(OutputPlane(z-step)) #creates output plane at 
                #the paraxial focus and propagates the bundle.
                return z-step #previous z location is the paraxial focus.
        
                
class ThickLens(OpticalElement):
    """ A lens with two optical element faces which take parameters c1 and c2.
    element centred around centre, with thickness d and refractive indices n1 
    and n2 (inside and out repsectively)"""
    
    def __init__(self, c1, c2, d, n1=1, n2=1.5168, centre=100, r_aper = 25):
        self.c1 = c1 #most variables kept public so that some aspects can
        #be easily changed without instantiating a new lens for ease.
        self.c2 = c2
        if self.c1 != 0:
            self.R1 = abs(1/self.c1)
        else:
            self.R1 = None
        if self.c2 != 0:
            self.R2 = abs(1/self.c2)
        else:
            self.R2 = None
        self.d = d
        self.n1 = n1
        self.n2 = n2
        self._centre = centre
        self.r_aper = r_aper
        
        self.RMS3mm = 0
        self.RMS25mm = 0
        
        #surfaces are mangled as all attributes can be changed via the initial
        #variables. Any messing with the surfaces without changing the initial
        #variables could cause confusion.
        
        #calculate aperture radius for each lens using r**2 = (d/4)(d-4R)
        if self.R1 is not None:
            if self.d/2 >= self.R1:
                r_aper_1 = self.r_aper
            else:
                r_aper_1 = np.sqrt(self.d*self.R1 - self.d**2/4)
        else:
            r_aper_1 = self.R2
            
        if self.R2 is not None:    
            if self.d/2 >= self.R2:
                r_aper_2 = self.r_aper
            else:
                r_aper_2 = np.sqrt(self.d*self.R2 - self.d**2/4)
        else:
            r_aper_2 = self.R1
        
        self.__surf_1 = SphericalRefraction(self._centre - self.d/2, self.n1, 
                                            self.n2, self.c1, r_aper_1)
        self.__surf_2 = SphericalRefraction(self._centre + self.d/2, self.n2, 
                                            self.n1, self.c2, r_aper_2)
        return None
    
    def propagate_ray(self, ray):
        """Implements OpticalElement method propagate_ray() by propagating a 
        ray individually through each spherical surface."""
        self.__surf_1.propagate_ray(ray)
        self.__surf_2.propagate_ray(ray)
        return self
    
    def focal_plane_est(self):
        """Estimates the z-axis position of the paraxial focus of the lens by 
        propagating a single ray close to the optical axis and finds the point 
        where the ray intercepts the axis."""
        testray = Ray([0,0.01,0], [0,0,1])
        self.propagate_ray(testray)
        test_k = testray.k()[1] 
        t = -testray.p()[1]/test_k
        pos = testray.p()[2] + t*testray.k()[2]
        if pos < self._centre + self.d/2:
            return None
        else:
            return pos
    
    def focal_length_est(self):
        """Estimates the focal length of the lens from the position of the 
        paraxial focus and the calculation of h_2 (distance between rear 
        intercept and the secondary principle plane). For details see Lensmaker
        Equation for thick lens"""
        f_p = self.focal_plane_est()
        if f_p is not None:
            i = f_p - (self._centre + self.d/2)
            f = i/(1- (self.n2-self.n1)*self.d*self.c1/self.n2)
            if f<0:
                return None
            return f
        else:
            return None
        
    def plot_f_d(self, d_min, d_max, step):
        """Plots focal length of lens against lens thickness, for lens thickness
        d_min <= d < d_max with 'step' spacing between."""
        
        d_arr = np.arange(d_min, d_max, step)
        f = [] 
        i=0
        check = len(d_arr)-1
        while i<= check:
            temp_lens = ThickLens(self.c1, self.c2, d_arr[i])
            f_est = temp_lens.focal_length_est()
            if f_est == None:
                d_arr = np.delete(d_arr, i)
                check = len(d_arr)-1
            else:
                f.append(f_est)
                i += 1
                
        #Plots graph - commented out so different graph can be plot.
#        fig1 = plt.figure(1)
#        plt.grid()
#        plt.plot(d_arr, f, linewidth = 3, color = 'indigo')
#        plt.xlabel("Lens Thickness (mm)", fontsize = 15, **csfont)  
#        plt.ylabel("Focal Length (mm)", fontsize = 15, **csfont)  
        
        return [d_arr, f]
    
    def plot_lensmaker(self, d_min, d_max, step):
        """
        """
        d_arr = np.arange(d_min, d_max, step)
        f_lm = []
        plt.grid()
        n = self.n2-self.n1
        for i in d_arr:
           inv_f = n*(self.c1-self.c2 + (n*i*self.c1*self.c2)/self.n2)
           f_lm.append(1/inv_f)
           #uncomment to plot in method
#        plt.plot(d_arr, f_lm, linewidth = 3, color = 'darkslategrey')
#        plt.xlabel("Lens Thickness (mm)", fontsize = 15, **csfont)  
#        plt.ylabel("Focal Length from Calculation (mm)", fontsize = 15, **csfont)   
#        plt.show()       
        return [d_arr, f_lm]

    
    def plot_RMS_d(self, d_min, d_max, step):
        """Plots a graph of the RMS spot radius at the focal plane of lenses of
        thicknesses between d_min and d_max (separated by step)."""
        
        d_arr = np.arange(d_min, d_max, step)
        RMS_arr = []
        i = 0
        check = len(d_arr) - 1
        while i <= check:
            bundle = RayBundle(5, 2)
            temp_lens = ThickLens(self.c1, self.c2, d_arr[i])
            f_est = temp_lens.focal_plane_est()
            if f_est == None:
                d_arr = np.delete(d_arr, i)
                check = len(d_arr)-1

            else:
                out = OutputPlane(f_est)
                bundle.propagate(temp_lens)
                bundle.propagate(out)
                RMS_arr.append(bundle.RMS())
                if i == 3:
                    self.RMS3mm = bundle.RMS()
                elif i == 25:
                    self.RMS25mm = bundle.RMS()
                i += 1
        return [d_arr, RMS_arr]
    ### commented to be plotted in a different format separately ###
#        fig2 = plt.figure(2)
#        plt.grid()
#        plt.rcParams["figure.figsize"] = [8,6]
#        plt.plot(d_arr, RMS_arr, linewidth = 3, color = 'indigo')
#        plt.xlabel("Lens Thickness (mm)", fontsize = 15, **csfont)  
#        plt.ylabel("RMS Spot Radius at Focal Plane (mm)", fontsize = 15, **csfont)  
#        plt.show()
#        return None
    
    def getRMS3mm(self):
        """
        Retrieves RMS Spot Radius at 3mm from the lens. Introduced at the end 
        to attain data.
        """
        return self.RMS3mm
    
    def getRMS25mm(self):
        """
        Retrieves RMS Spot Radius at 3mm from the lens. Introduced at the end 
        to attain data.
        """
        return self.RMS25mm
    
    def RMS_for_COLC(self, COLC_pos):
        """Returns RMS spot radius of a standard ray bundle (radius = 5, rings = 5)
        at the position of a plane at COLC_pos. For use in optimising to find the 
        position of the circle of least confusion.
        
        NOT USED IN FINAL REPORT, LEFT IN FOR COMPLETENESS"""
        bundle = RayBundle(5,5)
        out = OutputPlane(COLC_pos)
        bundle.propagate(self)
        bundle.propagate(out)
        return bundle.RMS()
        
    def focal_plane_est_opt(self):
        """Estimates the z-axis position of the circle of least confusion by 
        optimising RMS_for_COLC to a ray bundle
        
        NOT USED IN FINAL REPORT, LEFT IN FOR COMPLETENESS"""
        opt = fmin_tnc(self.RMS_for_COLC, [200], approx_grad=True)
        return opt[0]
    
    def focal_length_est_opt(self):
        """Estimates the focal length of the lens from the position of the 
        centre of least confusion, calculated from optimising RMS_for_COLC, and 
        the calculation of h_2 (distance between rear intercept and the secondary 
        principle plane). For details see Lensmaker Equation for thick lens.
        
        NOT USED IN FINAL REPORT, LEFT IN FOR COMPLETENESS"""
        f_p = self.focal_plane_est_opt()
        f_p = f_p[0]
        if f_p is not None:
            i = f_p - (self._centre + self.d/2)
            f = i/(1- (self.n2-self.n1)*self.d*self.c1/self.n2)
            if f<0:
                return None
            return f
        else:
            return None
        
    def plot_f_opt_d(self, d_min, d_max, step):
        """Plots focal length of lens against lens thickness, for lens thickness
        d_min <= d < d_max with 'step' spacing between.
        
        ***Did not produce useful results***
        NOT USED IN FINAL REPORT, LEFT IN FOR COMPLETENESS"""
        
        d_arr = np.arange(d_min, d_max, step)
        f = [] 
        i=0
        check = len(d_arr)-1
        while i<= check:
            f_est = self.focal_length_est_opt()
            if f_est == None:
                d_arr = np.delete(d_arr, i)
                check = len(d_arr)-1
            else:
                f.append(f_est)
                i += 1
        return [d_arr, f]
        
    def plot_RMS_opt_d(self, d_min, d_max, step):
        """Plots a graph of the RMS spot radius at the plane of the centre of 
        least confusion for lenses ofthicknesses between d_min and d_max 
        (separated by step).
        
        ***Did not produce useful results***
        NOT USED IN FINAL REPORT, LEFT IN FOR COMPLETENESS"""
        
        d_arr = np.arange(d_min, d_max, step)
        RMS_arr = []
        i = 0
        check = len(d_arr) - 1
        while i <= check:
            bundle = RayBundle(5, 2)
            temp_lens = ThickLens(self.c1, self.c2, d_arr[i])
            f_est = temp_lens.focal_plane_est()
            if f_est == None:
                d_arr = np.delete(d_arr, i)
                check = len(d_arr)-1

            else:
                out = OutputPlane(f_est)
                bundle.propagate(temp_lens)
                bundle.propagate(out)
                RMS_arr.append(bundle.RMS())
                i += 1
    
        return [d_arr, RMS_arr]
    
def RMS_from_curvature(c_arr):
    """Returns RMS spot radius of a standard ray bundle (radius = 5, rings = 5) 
    at the global variable fixed_plane, given the guess parameters for
    curvature in a list [c1, c2]. Used in Optimisation."""
    global fixed_thickness
    global fixed_plane
    c1 = c_arr[0]
    c2 = c_arr[1]
    lens = ThickLens(c1, c2, fixed_thickness)
    focal_plane = OutputPlane(fixed_plane)
    
    std_bundle = RayBundle(5,5)
    std_bundle.propagate(lens)
    std_bundle.propagate(focal_plane)
    
    return std_bundle.RMS()

def RMS_from_c2(c2_list):
    """Returns RMS spot radius of a standard ray bundle (radius = 5, rings = 5) 
    at the global variable fixed_plane, given the guess parameter for second
    curvature in a list [c2]. Used in Optimisation."""
    global fixed_thickness
    global fixed_plane
    c2 = c2_list[0]
    lens = ThickLens(0, c2, fixed_thickness)
    focal_plane = OutputPlane(fixed_plane)
    
    std_bundle = RayBundle(5,5)
    std_bundle.propagate(lens)
    std_bundle.propagate(focal_plane)
    
    return std_bundle.RMS()

def RMS_from_thickness(d_list):
    """Returns RMS spot radius of a standard ray bundle (radius = 5, rings = 5) 
    at the global variable fixed_plane, given the guess parameter for
    thickness in a list [d]. Used in Optimisation."""
    global fixed_plane
    global fixed_curvs
    d = d_list[0]
    c1 = fixed_curvs[0]
    c2 = fixed_curvs[1]
    lens = ThickLens(c1, c2, d)
    focal_plane = OutputPlane(fixed_plane)
    std_bundle = RayBundle(5,5)
    std_bundle.propagate(lens)
    std_bundle.propagate(focal_plane)
    
    return std_bundle.RMS()

def RMS_both_c_and_d(c_d_arr):
    """Returns RMS spot radius of a standard ray bundle (radius = 5, rings = 5)
    at the global variable fixed_plane, given the guess parameter [c1, c2, d].
    Used in optimisation"""
    global fixed_plane
    c1 = c_d_arr[0]
    c2 = c_d_arr[1]
    d = c_d_arr[2]
    lens = ThickLens(c1, c2, d)
    focal_plane = OutputPlane(fixed_plane)
    std_bundle = RayBundle(5,5)
    std_bundle.propagate(lens)
    std_bundle.propagate(focal_plane)
    
    return std_bundle.RMS()

def optimise_curvature_fixed_f(initial_guess, focal_plane=None):
    """Optimises the function RMS_from_curvature to return the two curvatures
    [c1, c2] which yield the smallest RMS spot radius for a given focal plane 
    position and a fixed thickness d=5mm. Default plane at 200mm."""
    
    if focal_plane is not None:
        global fixed_plane
        fixed_plane = focal_plane
        
    opt = fmin_tnc(RMS_from_curvature, initial_guess, approx_grad=True)
    c1 = opt[0][0]
    c2 = opt[0][1]
    return [c1, c2]

def optimise_c2_fixed_f(initial_guess, focal_plane=None):
    """Optimises the function RMS_from_curvature to return the two curvatures
    [c1, c2] which yield the smallest RMS spot radius for a given focal plane 
    position and a fixed thickness d=5mm. Default plane at 200mm."""
    
    if focal_plane is not None:
        global fixed_plane
        fixed_plane = focal_plane
        
    opt = fmin_tnc(RMS_from_c2, initial_guess, approx_grad=True)
    c2 = opt[0][0]
    return c2

def optimise_thickness_fixed_c(initial_guess, focal_plane=None):
    """Optimises the function RMS_from_thickness to return the thickness [d]
    which yields the smallest spot RMS for a given focal plane position
    and fixed curvatures (from optimisation or otherwise). Default plane at 200mm."""
    
    if focal_plane is not None:
        global fixed_plane
        fixed_plane = focal_plane
    opt = fmin_tnc(RMS_from_thickness, initial_guess, approx_grad=True)
    d = opt[0]
    return d

def optimise_thickness_curvature(initial_guess, focal_plane=None):
    """Optimises the function RMS_both_c_and_d to return optimised parameters
    [c1,c2,d] which yield the smallest RMS spot radius for a given focal plane 
    position. Default plane at 200mm."""
    
    if focal_plane is not None:
        global fixed_plane
        fixed_plane = focal_plane
    opt = fmin_tnc(RMS_from_thickness, initial_guess, approx_grad=True)
    c1 = opt[0][0]
    c2 = opt[0][1]
    d = opt[1]
    return [c1,c2,d]
    
To get started, import the module raytracer. In the file Lens_Thickness_Investigation
you can see I have called it 'rt'.

All the classes and methods are documented. To generate a lens, use
lens_name = rt.ThickLens(c1, c2, thickness). There are other arguments but they 
all default to useful values. See more in documentation.

To generate a ray bundle, use ray_bundle_name = rt.RayBundle(bundle_radius,
								number_rings).
The propagate the ray bundle through the lends simply use 
ray_bundle_name.propagate(lens_name).

That should get you started.
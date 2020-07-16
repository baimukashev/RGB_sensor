Labels in the dataset is stored as follows:

image_name, contact_type, x,y,Fn, Fx, Fy, Fz, Tz

image_name - unique image name
contact_type - 0/1 (0 - normal, 1 - shear/torsion)   

For normal test:

x - Cartesian x coordinate of end effector
y - Cartesian y coordinate of end effector
Fn - Normal force during normal test


For shear/torsion experiment:

Fx - force along z axis
Fy - force along x axis
Fz - force along y axis
Tz - torque around z axis

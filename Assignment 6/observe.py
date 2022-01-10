import numpy as np
from color_histogram import color_histogram
from chi2_cost import chi2_cost

def observe(particles, frame, bbox_height, bbox_width, hist_bin, hist, sigma_observe):
    
    xmin_particles = particles[:,0]-bbox_width/2
    ymin_particles = particles[:,1]-bbox_height/2
    
    xmax_particles = particles[:,0]+bbox_width/2
    ymax_particles = particles[:,1]+bbox_height/2
    
    particles_w = np.zeros((len(particles),1))
    
    for i in range(len(particles)):
        hist_particle_i = color_histogram(xmin_particles[i], 
                                          ymin_particles[i], 
                                          xmax_particles[i], 
                                          ymax_particles[i], 
                                          frame, 
                                          hist_bin)
        chi = chi2_cost(hist_particle_i, hist)
        
        if 0.5*((chi/sigma_observe)**2)>12:
            particles_w[i] = 0
        else:
            particles_w[i] = (1/(np.sqrt(2*np.pi)*sigma_observe))*np.exp(-0.5*(chi/sigma_observe)**2)
    
    
    particles_w = particles_w/np.sum(particles_w)
    
    return particles_w
    
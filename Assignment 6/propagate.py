import numpy as np

def propagate(particles, frame_height, frame_width, params):
    
    if params["model"]==0:
        A = np.eye(2)
        
        temp = params["sigma_position"]
        
        noise_std = np.array([temp,temp])
    
    if params["model"]==1:
        A = np.array([[1,0,1,0],[0,1,0,1],[0,0,1,0],[0,0,0,1]])
        temp1 = params["sigma_position"]
        temp2 = params["sigma_velocity"]
        noise_std = np.array([temp1,temp1,temp2,temp2])
        
    noise_particles = noise_std*np.random.randn(particles.shape[0],particles.shape[1])
    
    particles_new = (A@particles.T).T+noise_particles
    
    particles_new[:,0] = np.clip(particles_new[:,0], 0, frame_width-1)
    particles_new[:,1] = np.clip(particles_new[:,1], 0, frame_height-1)
    
    return particles_new
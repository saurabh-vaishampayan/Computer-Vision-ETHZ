import numpy as np

def resample(particles,particles_w):
    
    particles_resampled = np.zeros(particles.shape)
    particles_w_resampled = np.zeros(particles_w.shape)
    
    N_particles = len(particles)
    
    r = np.random.rand(1)/N_particles
    
    c = particles_w[0]
    
    i = 0
    
    for n in range(N_particles):
        U = r+((n-1)/N_particles)
        while U>c:
            i = i+1
            if i>=N_particles:
                i = 0
            
            c = c+particles_w[i]
        
        particles_resampled[n] = particles[i].copy()
        particles_w_resampled[n] = particles_w[i]
    
    particles_w_resampled = particles_w_resampled/np.sum(particles_w_resampled)
    
    return particles_resampled, particles_w_resampled


def resample_old(particles, particles_w):
    
    idx = np.random.multinomial(n=1,pvals=particles_w[:,0],size=len(particles))
    sampled_idx = np.argmax(idx,axis=1)
    
    particles_resampled = particles[sampled_idx]
    particles_w_resampled = particles_w[sampled_idx]
    
    particles_w_resampled = particles_w_resampled/np.sum(particles_w_resampled)
    
    return particles_resampled, particles_w_resampled
    
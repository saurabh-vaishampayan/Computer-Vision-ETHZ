import numpy as np


def color_histogram(xmin,ymin,xmax,ymax,frame,hist_bin):
    
    frame_shape = frame.shape
    
    x_min_ = max(0,round(xmin))
    y_min_ = max(0,round(ymin))
    
    x_max_ = min(frame_shape[1]-1,round(xmax))
    y_max_ = min(frame_shape[0]-1,round(ymax))
    
    hist_color = np.zeros((3,hist_bin))
    
    hist_color[0], _ = np.histogram(frame[y_min_:y_max_,x_min_:x_max_,0],hist_bin,range=[0,255])
    hist_color[1], _ = np.histogram(frame[y_min_:y_max_,x_min_:x_max_,1],hist_bin,range=[0,255])
    hist_color[2], _ = np.histogram(frame[y_min_:y_max_,x_min_:x_max_,2],hist_bin,range=[0,255])
    
    hist_color = np.ravel(hist_color)
    
    hist_color = hist_color/np.sum(hist_color);
    
    return hist_color
    
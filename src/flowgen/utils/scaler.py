import torch

class FeatureScaler(torch.nn.Module):
    def __init__(self, n_features):
        super().__init__()
        self.mean = None
        self.var = None
        self.N = None
        self.n_features = n_features
    
    def fit(self, data):
        # Stack features from all graphs
        b, c, nx, ny, nz, T = data.shape
        features = data.reshape(b,c,-1)
        self.mean = features.mean(dim=(0,2))
        self.var = features.var(dim=(0,2))

        self.N = b * T * nx * ny * nz
        
    @torch.no_grad()
    def forward(self, data):
        device = data.device
        b, c = data.shape[:2]
        T = data.shape[-1]
        with torch.autocast(device_type=device.type):
            mean = self.mean.to(device)
            std = self.var**0.5
            std = std.to(device)
            # Scale features
            data = (data - mean[None,:,None,None,None,None]) / std[None,:,None,None,None,None] 
            
            # min_max instance scaling
            #self.max = torch.zeros(b,c,T, device=device)
            #self.min = torch.zeros(b,c,T, device=device)

            #for i in range(b):
            #    for j in range(T):
            #        self.max[i,:,j], _ = torch.max(data[i,:,...,j].flatten(1,3), dim=1)
            #        self.min[i,:,j], _ = torch.min(data[i,:,...,j].flatten(1,3), dim=1)
            
            #data = 2 * (data - self.min[:,:,None,None,None,:])/(self.max[:,:,None,None,None,:] - self.min[:,:,None,None,None,:]) - 1
            
        
        return data
        
    @torch.no_grad()
    def denorm(self, data):
        device = data.device
        mean = self.mean.to(device)
        std = self.var**0.5
        std = std.to(device)

        #minmax_denorm
        #data =  (data + 1)/2 * (self.max[:,:,None,None,None,:] - self.min[:,:,None,None,None,:]) + self.min[:,:,None,None,None,:]

        #standard denorm
        data = data  * std[None,:,None,None,None,None]  + mean[None,:,None,None,None,None] 

        return data
        

    def update_stats(self, data):
        device = data.device
        mean = self.mean.to(device)
        var = self.var.to(device)
        b, c, nx, ny, nz, T = data.shape
        features = data.reshape(b,c,-1)
        
        new_mean = features.mean(dim=(0,2))
        new_var = features.var(dim=(0,2))
        #print(f"Prev std: {var**0.5}")
        #print(f"New std: {new_var**0.5}")
        N_new = b * T * nx * ny * nz

        self.mean = self.N/(self.N + N_new) * mean + N_new/(self.N + N_new) * new_mean

        n = self.N + N_new

        self.var = 1/(n-1) * ((N_new-1)*new_var + (self.N-1)*var + (new_mean - mean)**2 * self.N*N_new / n)
        self.N = n
        #print(f"Updated std: {self.var**0.5}")
import torch 


class PolarCoordinateEmbedding(torch.nn.Module):
    """
    Provides basic polar coordinates embedding functionality in the form of PyTorch module
    """
    def __init__(self, output_dim):
        super(PolarCoordinateEmbedding, self).__init__()
        self.output_dim = output_dim
        self.radius_embedding = torch.nn.Linear(1,output_dim)
        self.angle_embedding = torch.nn.Linear(1,output_dim)
        print("Output dim PolarCoordinateEmbedding:" + str(output_dim))

    def forward(self, x):
        longitude = x[...,0]
        latitude = x[...,1]
        
        #calculate radius as the euclidean distance from the origin
        radius = torch.sqrt(longitude**2 + latitude**2)

        #calculate angle using atan2

        angle = torch.atan2(latitude, longitude)
        #angle = torch.where(radius == 0, torch.tensor(float('nan')),
                            #torch.where(latitude >= 0, torch.acos(longitude / radius),
                                        #-torch.acos(longitude / radius)))

        # Embed radius and angle separately
        radius_embedded = self.radius_embedding(radius.unsqueeze(-1))
        angle_embedded = self.radius_embedding(angle.unsqueeze(-1))

        #concatenate radis and angle embeddings along the last dimension
        return torch.cat((radius_embedded, angle_embedded),dim=-1)

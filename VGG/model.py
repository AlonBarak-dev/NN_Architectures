from torch import nn


class Print(nn.Module):
    def __init__(self):
        super(Print, self).__init__()

    def forward(self, x):
        print(x.shape)
        return x


class VGG(nn.Module):

    def __init__(self, model:str="vgg-16"):
        super().__init__()
        self.vgg = None
        self.activation = nn.ReLU()
        if model == "vgg-19":
            self.vgg = self.create_vgg19()
        else:
            self.vgg = self.create_vgg16()

    def forward(self, X):
        logits = self.vgg(X)
        return logits

    
    def create_vgg(self, layers):
        """
            This method init a VGG-16 network.
            It return the final model to the init function
        """
        # structure of the network in terms of channels per block
        channels = [3, 64, 128, 256, 512, 512]
        # init a sequential model to hold the blocks
        model = nn.Sequential()
        # init default params
        conv_params = {
            'in_dim': None,
            'out_dim': None,
            'kernel_size':(3, 3),
            'stride': 1, 
            'padding': 0, 
            'layers': None
        }
        pool_params = {
            'stride': 1, 
            'pool_size': (2, 2)
        }
        
        # create conv_pool blocks
        for i in range(len(channels)-1):
            conv_params['layers'] = layers[i]
            conv_params['in_dim'], conv_params['out_dim'] = channels[i], channels[i+1]
            conv_block = self.Conv_pool_block(conv_params, pool_params)
            model.append(conv_block)

        # flatten the image before FC head
        model.append(nn.Flatten())
        
        # TBD features shapes - per task
        head = self.Fc_head(4608, 1000)
        model.append(head)
        return model


    def create_vgg16(self):
        return self.create_vgg([2, 2, 2, 3, 3])
    
    def create_vgg19(self):
        return self.create_vgg([2, 2, 2, 4, 4])

    
    def Conv_pool_block(self, conv_params:dict, pool_params:dict):
        # extract parameters for Conv. layers and Pool. layer
        in_dim, out_dim = conv_params['in_dim'], conv_params['out_dim']
        kernel_size, conv_stride, padding = conv_params['kernel_size'], conv_params['stride'], conv_params['padding']
        num_of_layers = conv_params['layers']
        pool_stride, pool_size = pool_params['stride'], pool_params['pool_size']
        
        # init a new sequential block
        conv_block = nn.Sequential()
        # add 'num_of_layers' convolutional layers to the block
        for idx in range(num_of_layers):
            conv = nn.Conv2d(in_dim, out_dim, kernel_size, conv_stride, padding)
            conv_block.append(conv)
            conv_block.append(nn.ReLU())
            in_dim = out_dim
            
        
        # add a pooling layer
        pool_layer = nn.MaxPool2d(pool_size, pool_stride)
        conv_block.append(pool_layer)

        return conv_block

    def Fc_head(self, in_features, out_features):
        """
            This method create a FC head suits for both 16 & 19 VGGs.
        """
        fc1 = nn.Linear(in_features, 4096)
        fc2 = nn.Linear(4096, 4096)
        fc3 = nn.Linear(4096, out_features)
        head = nn.Sequential(fc1, nn.ReLU(), fc2, nn.ReLU() , fc3)
        return head
        
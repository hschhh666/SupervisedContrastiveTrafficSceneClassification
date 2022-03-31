import torch
import torch.nn as nn
from torchvision.models import resnet50, resnet18

class myResnet50(nn.Module):
    def __init__(self, feat_dim = 4, pretrained = False, parallel = True):
        super(myResnet50, self).__init__()
        self.model = resnet50(num_classes=365)
        places365_pre_trained_model_file = '/home/hsc/Research/TrafficSceneClassification/code/testExperiment/places365PreTrained/resnet50_places365.pth.tar'
        checkpoint = torch.load(places365_pre_trained_model_file, map_location=lambda storage, loc: storage)
        state_dict = {str.replace(k,'module.',''): v for k,v in checkpoint['state_dict'].items()}
        self.model.load_state_dict(state_dict)
        
        self.feat_dim_before_fc = self.model.fc.in_features
        self.feat_dim_after_fc = feat_dim
        self.model.fc = self._build_mlp(2, self.model.fc.in_features, 128, self.feat_dim_after_fc)
        if parallel:
            self.model = nn.DataParallel(self.model)

    def _build_mlp(self, num_layers, input_dim, mlp_dim, output_dim, last_bn=True):
        mlp = []
        for l in range(num_layers):
            dim1 = input_dim if l == 0 else mlp_dim
            dim2 = output_dim if l == num_layers - 1 else mlp_dim

            mlp.append(nn.Linear(dim1, dim2, bias=False))

            if l < num_layers - 1:
                mlp.append(nn.BatchNorm1d(dim2))
                mlp.append(nn.ReLU(inplace=True))
            elif last_bn:
                # follow SimCLR's design: https://github.com/google-research/simclr/blob/master/model_util.py#L157
                # for simplicity, we further removed gamma in BN
                mlp.append(nn.BatchNorm1d(dim2, affine=False))
        return nn.Sequential(*mlp)
    
    def forward(self, x):
        return nn.functional.normalize(self.model(x), dim=1)


class myResnet18(nn.Module):
    def __init__(self, feat_dim = 4, pretrained = False, parallel = True):
        super(myResnet18, self).__init__()
        self.model = resnet18(pretrained=pretrained)
        self.feat_dim_before_fc = self.model.fc.in_features
        self.feat_dim_after_fc = feat_dim
        self.model.fc = self._build_mlp(2, self.model.fc.in_features, 128, self.feat_dim_after_fc)
        if parallel:
            self.model = nn.DataParallel(self.model)

    def _build_mlp(self, num_layers, input_dim, mlp_dim, output_dim, last_bn=True):
        mlp = []
        for l in range(num_layers):
            dim1 = input_dim if l == 0 else mlp_dim
            dim2 = output_dim if l == num_layers - 1 else mlp_dim

            mlp.append(nn.Linear(dim1, dim2, bias=False))

            if l < num_layers - 1:
                mlp.append(nn.BatchNorm1d(dim2))
                mlp.append(nn.ReLU(inplace=True))
            elif last_bn:
                # follow SimCLR's design: https://github.com/google-research/simclr/blob/master/model_util.py#L157
                # for simplicity, we further removed gamma in BN
                mlp.append(nn.BatchNorm1d(dim2, affine=False))
        return nn.Sequential(*mlp)
    
    def forward(self, x): # 注意，该网络输出的特征已经归一化了
        return nn.functional.normalize(self.model(x), dim=1)



if __name__ == '__main__':
    x = torch.rand([1,3,400,224])
    model = myResnet18()
    print(model(x).shape)
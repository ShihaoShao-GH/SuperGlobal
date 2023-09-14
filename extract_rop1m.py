import torch
from model.CVNet_Rerank_model import CVNet_Rerank
from test.dataset_1m import DataSet
from tqdm import tqdm
import torch.nn.functional as F
import argparse

@torch.no_grad()
def extract_feature(model, test_loader, scale_list):
    with torch.no_grad():
        
        img_feats = [[] for i in range(1)] 
        count = 0
        for im_list in tqdm(test_loader):
            if count % 10000 == 0:
                print(f"Image Processed {count}")
            count+=1
            
            for idx in range(len(im_list)):
                im_list[idx] = im_list[idx].cuda()
                desc = model.extract_global_descriptor(im_list[idx], True, True, True, scale_list)
                if len(desc.shape) == 1:
                    desc.unsqueeze_(0)
                desc = F.normalize(desc, p=2, dim=1)
                img_feats[idx].append(desc.detach().cpu())
            
        for idx in range(len(img_feats)):
            img_feats[idx] = torch.cat(img_feats[idx], dim=0)
            if len(img_feats[idx].shape) == 1:
                img_feats[idx].unsqueeze_(0)

        img_feats_agg = F.normalize(torch.mean(torch.cat([img_feat.unsqueeze(0) for img_feat in img_feats], dim=0), dim=0), p=2, dim=1)
        

    return img_feats_agg





def main():
    parser = argparse.ArgumentParser(description='Generate 1M embedding')
    parser.add_argument('--weight',
                        help='Path to weight')
    parser.add_argument('--depth', default=101, type=int,
                        help='Depth of ResNet')
    args = parser.parse_args()
    weight_path, depth =  args.weight,  args.depth
    model1 = CVNet_Rerank(depth, 2048, True)
        
    weight = torch.load(weight_path)
    weight_new = {}
    for i,j in zip(weight['model_state'].keys(), weight['model_state'].values()):
            weight_new[i.replace('globalmodel','encoder_q')] = j
            
    mis_key = model1.load_state_dict(weight_new, strict=False)
    
    model1.cuda()
    
    dataset = DataSet("/data1/shaoshihao/rop1m/RevistedOP")
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        sampler=None,
        num_workers=4,
        pin_memory=True,
        drop_last=False,
    )
    features = extract_feature(model1, dataloader, 3)
    torch.save(features, f"feats_1m_RN{depth}.pth")
if __name__ == "__main__":
    main()
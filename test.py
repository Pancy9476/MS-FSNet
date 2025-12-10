import torch
import os
# from model import MultiScaleTransformerFusion
from model_mamba_highlow_visual1 import MultiScaleMambaFusion
# from model_mamba_highlow_vl import MultiScaleMambaFusion
from dataset import MyFusionDataset
from torch.utils.data import DataLoader
from tqdm import tqdm
import argparse
import torchvision.utils as vutils
import time
@torch.no_grad()
def tt(model, dataloader, device, save_dir):
    model.eval()
    # total_params = (sum(p.numel() for p in model.parameters())) / 1e6
    # print(f"Total parameters: {total_params :.6f}M")  # 输出以百万为单位
    st=time.time()
    os.makedirs(save_dir, exist_ok=True)
    for idx, (pan, ms, gt, name) in enumerate(tqdm(dataloader, desc="Testing")):
        pan, ms = pan.to(device), ms.to(device)
        out, _ = model(pan, ms)
        out = out.clamp(0, 1)

        # ==== 可视化特征图 ==== 灰度
        # os.makedirs(f"vis/fly", exist_ok=True)
        # with torch.no_grad():
        #     import numpy as np
        #     import matplotlib.pyplot as plt
        #     import matplotlib.cm as cm
        #     from PIL import Image
        #     # 傅里叶变换 + 频谱居中
        #     freq = torch.fft.fftshift(torch.fft.fft2(gt[0]), dim=(-2, -1))
        #     # freq = torch.fft.fft2(gt[0])
        #     amp = torch.log1p(torch.abs(freq)).cpu().numpy()  # (3, H, W)
        #     rgb_img = np.zeros((amp.shape[1], amp.shape[2], 3), dtype=np.uint8)  # HWC
        #     # 获取 colormap 对象
        #     colormap = cm.get_cmap('viridis')
        #     for i in range(3):
        #         ch = amp[i]
        #         ch = (ch - ch.min()) / (ch.max() - ch.min() + 1e-8)  # 归一化到 [0,1]
        #         # 使用 colormap，将灰度图映射为 RGBA，再取 RGB
        #         colored = colormap(ch)[:, :, :3]  # shape: (H, W, 3)
        #         rgb_img[:, :, i] = (colored[:, :, i] * 255).astype(np.uint8)  # 每个通道分别给 RGB
        #         # 保存图像
        #     Image.fromarray(rgb_img).save(f"vis/fly/{idx}.png")

            # fly_np = fly.cpu().numpy()
            # # 分别保存三个通道的彩色频谱图
            # for i in range(3):
            #     ch = fly_np[i]
            #     ch = (ch - ch.min()) / (ch.max() - ch.min() + 1e-8)  # 归一化
            #     plt.imsave(f"vis/fly/{idx}.png",ch,cmap='viridis')
            # freq = torch.fft.fft2(gt)
            # fly = torch.abs(freq)
            # vis_img1 = torch.clamp(fly[0], min=0, max=1)
            # vis_img1 = vis_img1.permute(1, 2, 0).cpu().detach().numpy()
            # # vis_img1 = (vis_img1 - vis_img1.min()) / (vis_img1.max() - vis_img1.min() + 1e-8)
            # plt.imsave(f"vis/fly/{idx}.png", vis_img1, cmap='viridis')


        # Save result image
        filename = os.path.splitext(os.path.basename(name[0]))[0]
        vutils.save_image(out, os.path.join(save_dir, f"{filename}.png"))
        # vutils.save_image(gt, os.path.join(save_dir, f"gt_{idx}.png"))
    ed=time.time()
    print(f"Testing time: {(ed-st)/23}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--device', type=str, default='cuda:0' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--root_dir', type=str, default=r"datasets/Maryland")
    parser.add_argument('--model_path', type=str, default="checkpoints/Maryland/best_model.pth")  #best_model.pth  #epoch/epoch_10.pth
    parser.add_argument('--save_dir', type=str, default="test_results/Maryland")
    args = parser.parse_args()

    device = torch.device(args.device)

    test_set = MyFusionDataset(split='test', root_dir=args.root_dir)
    test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False, num_workers=4)

    # model = MultiScaleTransformerFusion().to(device)
    model = MultiScaleMambaFusion().to(device)
    model.load_state_dict(torch.load(args.model_path, map_location=device))

    # total_params = (sum(p.numel() for p in model.parameters())) / 1e6
    # print(f"Mamba Total params: {total_params :.6f}M")

    tt(model, test_loader, device, args.save_dir)

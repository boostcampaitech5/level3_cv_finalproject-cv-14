from .iresnet import iresnet18, iresnet34, iresnet50, iresnet100, iresnet200
from .mobilefacenet import get_mbf


def get_model(name, **kwargs):
    # resnet
    if name == "r18":
        return iresnet18(False, **kwargs)
    elif name == "r34":
        return iresnet34(False, **kwargs)
    elif name == "r50":
        return iresnet50(False, **kwargs)
    elif name == "r100":
        return iresnet100(False, **kwargs)
    elif name == "r200":
        return iresnet200(False, **kwargs)
    elif name == "r2060":
        from .iresnet2060 import iresnet2060
        return iresnet2060(False, **kwargs)

    elif name == "mbf":
        fp16 = kwargs.get("fp16", False)
        num_features = kwargs.get("num_features", 512)
        return get_mbf(fp16=fp16, num_features=num_features)

    elif name == "mbf_large":
        from .mobilefacenet import get_mbf_large
        fp16 = kwargs.get("fp16", False)
        num_features = kwargs.get("num_features", 512)
        return get_mbf_large(fp16=fp16, num_features=num_features)

    elif name == "vit_t":
        num_features = kwargs.get("num_features", 512)
        from .vit import VisionTransformer
        return VisionTransformer(
            img_size=112, patch_size=9, num_classes=num_features, embed_dim=256, depth=12,
            num_heads=8, drop_path_rate=0.1, norm_layer="ln", mask_ratio=0.1)

    elif name == "vit_t_dp005_mask0": # For WebFace42M
        num_features = kwargs.get("num_features", 512)
        from .vit import VisionTransformer
        return VisionTransformer(
            img_size=112, patch_size=9, num_classes=num_features, embed_dim=256, depth=12,
            num_heads=8, drop_path_rate=0.05, norm_layer="ln", mask_ratio=0.0)

    elif name == "vit_s":
        num_features = kwargs.get("num_features", 512)
        from .vit import VisionTransformer
        return VisionTransformer(
            img_size=112, patch_size=9, num_classes=num_features, embed_dim=512, depth=12,
            num_heads=8, drop_path_rate=0.1, norm_layer="ln", mask_ratio=0.1)
    
    elif name == "vit_s_dp005_mask_0":  # For WebFace42M
        num_features = kwargs.get("num_features", 512)
        from .vit import VisionTransformer
        return VisionTransformer(
            img_size=112, patch_size=9, num_classes=num_features, embed_dim=512, depth=12,
            num_heads=8, drop_path_rate=0.05, norm_layer="ln", mask_ratio=0.0)
    
    elif name == "vit_b":
        # this is a feature
        num_features = kwargs.get("num_features", 512)
        from .vit import VisionTransformer
        return VisionTransformer(
            img_size=112, patch_size=9, num_classes=num_features, embed_dim=512, depth=24,
            num_heads=8, drop_path_rate=0.1, norm_layer="ln", mask_ratio=0.1, using_checkpoint=True)

    elif name == "vit_b_dp005_mask_005":  # For WebFace42M
        # this is a feature
        num_features = kwargs.get("num_features", 512)
        from .vit import VisionTransformer
        return VisionTransformer(
            img_size=112, patch_size=9, num_classes=num_features, embed_dim=512, depth=24,
            num_heads=8, drop_path_rate=0.05, norm_layer="ln", mask_ratio=0.05, using_checkpoint=True)

    elif name == "vit_l_dp005_mask_005":  # For WebFace42M
        # this is a feature
        num_features = kwargs.get("num_features", 512)
        from .vit import VisionTransformer
        return VisionTransformer(
            img_size=112, patch_size=9, num_classes=num_features, embed_dim=768, depth=24,
            num_heads=8, drop_path_rate=0.05, norm_layer="ln", mask_ratio=0.05, using_checkpoint=True)
        
    elif name == "vit_h":  # For WebFace42M
        num_features = kwargs.get("num_features", 512)
        from .vit import VisionTransformer
        return VisionTransformer(
            img_size=112, patch_size=9, num_classes=num_features, embed_dim=1024, depth=48,
            num_heads=8, drop_path_rate=0.1, norm_layer="ln", mask_ratio=0, using_checkpoint=True)
    
    # ghostnet add
    elif name == 'ghost':
        import torch
        from .ghostnet import ghostnet
        ghostnet = ghostnet()
        
        # pretrained weights
        file_path = '/opt/ml/project/insightface/recognition/arcface_torch/pretrained/ghostnet/ghostnet_Final.pth'
        ghostnet.state_dict(torch.load(file_path))
        return ghostnet
    
    elif name =='mobile':
        import torch
        from .mobilenet import MobileNetV3

        mobilenet = MobileNetV3()
        file_path = '/opt/ml/project/insightface/recognition/arcface_torch/pretrained/mobilenet/mobilenet0.25_Final.pth'
        mobilenet.state_dict(torch.load(file_path))
        return mobilenet

    else:
        raise ValueError()
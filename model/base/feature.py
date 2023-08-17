r""" Extracts intermediate features from given backbone network & layer ids """
# Original code: HSNet (https://github.com/juhongm999/hsnet)

def extract_feat_res_pycls(img, backbone, feat_ids, bottleneck_ids, lids):
    r""" Extract intermediate features from ResNet"""
    feats = []

    # Layer 0
    feat = backbone.stem(img)

    # Layer 1-4
    for hid, (bid, lid) in enumerate(zip(bottleneck_ids, lids)):
        res = feat
        feat = backbone.__getattr__('s%d' % lid).__getattr__('b%d' % (bid+1)).f.forward(feat)

        if bid == 0:
            res = backbone.__getattr__('s%d' % lid).__getattr__('b%d' % (bid+1)).proj.forward(res)
            res = backbone.__getattr__('s%d' % lid).__getattr__('b%d' % (bid+1)).bn.forward(res)
        feat += res

        if hid + 1 in feat_ids:
            feats.append(feat.clone())

        feat = backbone.__getattr__('s%d' % lid).__getattr__('b%d' % (bid+1)).relu.forward(feat)

    return feats


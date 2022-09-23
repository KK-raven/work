from glob import glob
import pandas as pd
import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt


def _get_df(base_path='public-covid-data', folder='rp_im'):
    data_dict = pd.DataFrame({'FilePath': glob(f'{base_path}/{folder}/*'),
                              'FileName': [p.split('/')[-1] for p in glob(f'{base_path}/{folder}/*')]})
    return data_dict


def get_df_all(base_path='public-covid-data'):
    rp_im_df = _get_df(base_path, 'rp_im')
    rp_msk_df = _get_df(base_path, 'rp_msk')
    return rp_im_df.merge(rp_msk_df, on='FileName', suffixes=('Image', 'Mask'))


def load_nifti(path):
    nifti = nib.load(path)
    data = nifti.get_fdata()
    return np.rollaxis(data, axis=1, start=0)


def label_color(mask_volume,
                ggo_color=[255, 0, 0],
                consokidation_color=[0, 255, 0],
                effusion_color=[0, 0, 255]):
    # make a empty box for color
    shp = mask_volume.shape
    mask_color = np.zeros((shp[0], shp[1], shp[2], 3), dtype=np.float32)
    # asign the color
    mask_color[np.equal(mask_volume, 1)] = ggo_color
    mask_color[np.equal(mask_volume, 2)] = consokidation_color
    mask_color[np.equal(mask_volume, 3)] = effusion_color
    return mask_color


def hu_to_gray(volume):
    max_hu = np.max(volume)
    min_hu = np.min(volume)
    # range を 0~1 に変換する
    volume_rerange = (volume - min_hu) / max((max_hu - min_hu), 1e-3)
    # range　を 0~255 に変換する（この時点では float なので画像表示はうまくいかない）
    volume_rerange = volume_rerange * 255
    # shape をカラーの4次元に合わせるため、3枚重ね合わせて最後の axis に 3 が入るようにする
    volume_rerange = np.stack([volume_rerange, volume_rerange, volume_rerange], axis=-1)
    # このまま return すると float で画像がうまく表示されないため、0~255のintegerに変換する必要がある
    return volume_rerange.astype(np.uint8)


def overlay(gray_volume, mask_volume, mask_color, alpha=0.3):
    mask_filter = np.greater(mask_volume, 0)
    mask_filter = np.stack([mask_filter, mask_filter, mask_filter], axis=-1)
    overlayed = np.where(mask_filter,
                         ((1-alpha)*gray_volume + alpha*mask_color).astype(np.uint8),
                         gray_volume)
    
    return overlayed


def vis_overlay(overlayed, original_volume, mask_volume, cols=5, display_num=25, figsize=(15, 15)):
    rows = (display_num - 1) // cols + 1
    total_num = overlayed.shape[-2]
    interval = total_num / display_num
    if interval < 1:
        interval = 1
    fig, axes = plt.subplots(rows, cols, figsize=figsize)

    for i in range(display_num):
        row_i = i//cols
        col_i = i%cols
        idx = int(i * interval)  
        if idx >= total_num:
            break
        stats = get_hu_stats(original_volume[:, :, idx], mask_volume[:, :, idx])
        title = f'slice #: {idx}'
        title += f"\nggo mean: {stats['ggo_mean']:.0f}±{stats['ggo_std']:.0f}"
        title += f"\nconsolidation mean: {stats['consolidation_mean']:.0f}±{stats['consolidation_std']:.0f}"
        title += f"\neffusion mean: {stats['effusion_mean']:.0f}±{stats['effusion_std']:.0f}"
        axes[row_i, col_i].imshow(overlayed[:, :, idx])
        axes[row_i, col_i].set_title(title)
        axes[row_i, col_i].axis('off')
    fig.tight_layout()
        
        
def get_hu_stats(volume,
                 mask_volume,
                 label_dict={1: 'ggo', 2: 'consolidation', 3: 'effusion'}):

    result = {}

    for label in label_dict.keys():
        prefix = label_dict[label]
        roi_hu = volume[np.equal(mask_volume, label)]
        result[prefix + '_mean'] = np.mean(roi_hu)
        result[prefix + '_std'] = np.std(roi_hu)
    
    return result
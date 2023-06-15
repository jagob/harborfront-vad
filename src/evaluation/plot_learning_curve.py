# TODO plot learning rate
# TODO plot rbdc and tbdc
import os
import glob

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from tbparse import SummaryReader

from src.utils.config import load_config
from src.evaluation.evaluate import evaluate

sns.set_theme()
# sns.set_context("paper")


def load_auc_scores_mnad(cfg, dataset, model_dir, df_auc_scores):
    step = []
    auc_micros = []

    if 'mnad' in model_dir:
        epoch_dirs = sorted(glob.glob(os.path.join(model_dir, 'epoch_*')))
        epoch_dirs = [edir.replace(model_dir + '/', '') for edir in epoch_dirs]
    elif 'ssmtl' in model_dir:
        epoch_dirs_path = os.path.join(model_dir, 'per_epoch_predictions', dataset)
        epoch_dirs = sorted(os.listdir(epoch_dirs_path))

    for epoch_dir in epoch_dirs:
        auc_micro, auc_macro, *_ = evaluate(cfg, dataset, model_dir, epoch_dir, verbose=False)
        if auc_micro < 0:
            continue
        step.append(int(epoch_dir.replace('epoch_', '')))
        auc_micros.append(auc_micro)
    df = pd.DataFrame(list(zip(step, auc_micros)), columns=['step', 'auc_micro'])
    df['dataset'] = dataset
    df_auc_scores = pd.concat([df_auc_scores, df], ignore_index=True)
    return df_auc_scores


if __name__ == "__main__":
    cfg = load_config('config.yaml')

    # model_dir = os.path.join(cfg['models_path'], r'mnad/ucsdped2/pred/001')
    # model_dir = os.path.join(cfg['models_path'], r'mnad/avenue/pred/001')
    # model_dir = os.path.join(cfg['models_path'], r'mnad/harbor/recon/low_density_001')
    # model_dir = os.path.join(cfg['models_path'], r'mnad/harbor/recon/006')
    # model_dir = os.path.join(cfg['models_path'], r'mnad/harbor/pred/empty_001')
    # model_dir = os.path.join(cfg['models_path'], r'mnad/harbor/pred/low_density_001')
    # model_dir = os.path.join(cfg['models_path'], r'mnad/harbor/pred/002')
    # model_dir = os.path.join(cfg['models_path'], r'ssmtl++/harbor_low_density_3_task_gt')
    model_dir = os.path.join(cfg['models_path'], r'ssmtl++/harbor_low_density_3_task')

    test_sets = [
                 # 'ucsdped2',
                 # 'avenue',
                 # 'harbor-vehicles',
                 'harbor-appearance',
                 'harbor-fast-moving',
                 'harbor-near-edge',
                 'harbor-high-density',
                 'harbor-tampering',
                 # 'harbor-mannequin',
                 # 'harbor-realfall',
                ]

    reader = SummaryReader(model_dir)
    df = reader.scalars
    df = df.rename(columns={"step": "epoch", "tag": "loss"})
    if 'mnad' in model_dir:
        df = df.replace('model/train-recon-total-loss', 'train-loss')
        df = df.replace('model/val-recon-total-loss', 'val-loss')
        df = df.replace('model/train-pred-total-loss', 'train-loss')
        df = df.replace('model/val-pred-total-loss', 'val-loss')
        train_loss = df[df['loss'] == 'train-loss']
        val_loss = df[df['loss'] == 'val-loss']
        df_losses = pd.concat([train_loss, val_loss])
    if 'ssmtl' in model_dir:
        df = df.replace('train_loss/total', 'train-loss')
        df = df.replace('val_loss/total', 'val-loss')
        train_loss = df[df['loss'] == 'train-loss']
        val_loss = df[df['loss'] == 'val-loss']
        df_losses = pd.concat([train_loss, val_loss])

    df_auc_scores = pd.DataFrame({'step': pd.Series(dtype='int'),
                                  'auc_micro': pd.Series(dtype='float'),
                                  'dataset': pd.Series(dtype='str')})
    for test_set in test_sets:
        df_auc_scores = load_auc_scores_mnad(cfg, test_set, model_dir, df_auc_scores)
    df_auc_scores.dataset = df_auc_scores.dataset.str.replace('harbor-', '')
    if 'ssmtl' in model_dir:
        df_auc_scores.step += 1

    ax = sns.lineplot(data=df_losses, x='epoch', y='value', hue='loss',
                      # palette="tab10",
                      # palette="pastel",
                      palette="husl",
                      linewidth=1)
    ax.set(xlim=(0, df.epoch.max()))
    ax.set_xticks(range(df.epoch.max() + 1))
    ax.set(xlabel='Epoch')
    ax.set(ylabel='Loss')
    ax.yaxis.grid(False) # Hide the horizontal gridlines
    plt.legend(loc='upper left', title='Loss')

    ax2 = ax.twinx()
    sns.lineplot(data=df_auc_scores, x='step', y='auc_micro', hue='dataset',
                 palette="deep",
                 linewidth=2,
                 marker='o',
                 ax=ax2)
    ax2.set(ylabel='AUC')
    ax2.set(ylim=(0.4, 1.0))
    plt.legend(loc='upper right', title='Dataset AUC')

    plot_save_path = os.path.join(model_dir, 'learning_curve.png')
    plt.savefig(plot_save_path, bbox_inches='tight', pad_inches=0.1, dpi=100)
    plt.savefig(plot_save_path.replace('.png', '.pdf'), bbox_inches='tight', pad_inches=0.1)
    # plt.show()
    plt.close()

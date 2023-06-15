import os
import seaborn as sns
import matplotlib.pyplot as plt

from src.utils.config import load_config
from src.evaluation.evaluate import evaluate

# sns.set_theme()
sns.set_context("paper")

COLORS = sns.color_palette("deep", 8)
COLORS_JET = sns.color_palette("coolwarm_r", 11)


def plot_roc(plot_data, save_path):
    plt.figure()
    for fpr, tpr, label in plot_data:
        plt.plot(fpr, tpr, label=label)
    plt.plot([0, 1], [0, 1], linestyle='--', color='black')
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.grid()
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    # plt.title('Receiver operating characteristic')
    plt.legend(loc="lower right")
    plt.savefig(save_path, bbox_inches='tight', pad_inches=0.1, dpi=100)
    plt.savefig(save_path.replace('.png', '.pdf'), bbox_inches='tight', pad_inches=0.1) 
    # plt.show()
    plt.close()


if __name__ == "__main__":
    cfg = load_config('config.yaml')

    models = [['MNAD-Recon', r'mnad/harbor/recon/006', 'epoch_000020'],
              ['MNAD-Pred', r'mnad/harbor/pred/002', 'epoch_000020'],
              ['SSMTL', r'ssmtl++/harbor_low_density_3_task', '19'],
              ['PGM-Spatial', r'pgm/harbor/spatial/s01m04', None],]

    datasets = ['harbor-appearance', 'harbor-fast-moving', 'harbor-high-density', 'harbor-tampering']
    # datasets = ['harbor-appearance', 'harbor-fast-moving', 'harbor-near-edge', 'harbor-high-density', 'harbor-tampering']

    for dataset in datasets:
        plot_data = []
        for label, model, epoch_dir in models:
            model_dir = os.path.join(cfg['models_path'], model)
            ret = evaluate(cfg, dataset, model_dir, epoch_dir, verbose=False)
            auc_micro, auc_macro, fpr, tpr = ret
            plot_data.append([fpr, tpr, f'{label} AUC {auc_micro*100:.1f}'])

        save_path = f'src/data/tmp/img/roc-{dataset}.png'
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plot_roc(plot_data, save_path)

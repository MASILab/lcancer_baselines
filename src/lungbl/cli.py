import os, argparse, math, torch

import lightning as L
from lungbl.train import train, test, predict
from lungbl.config import LMODELS, Config, datamodule_from_config, trainer_from_config, checkpoint_from_config
from lungbl.utils.modeling import auc_from_df
from lungbl.cachedcohorts import NLST_CohortWrapper

from lungbl import definitions
from lungbl.utils.statutils import bootstrap
from lungbl.utils.tabular import read_tabular, format_datetime_str

def find_max_epoch(ckpt_folder):
    """
    Find the highest epoch in among saved checkpoints
    :param ckpt_folder: dir where the checkpoints are being saved.
    :return: Integer of the highest epoch reached by the checkpoints.
    """
    ckpt_files = os.listdir(ckpt_folder)  # list of strings
    epochs = [int(filename[6:-5]) for filename in ckpt_files]  # 'epoch={int}.ckpt' filename format
    max_idx = epochs.index(max(epochs)) # idnex of highest epoch
    return ckpt_files[max_idx]

COHORTS = {
    # Load cohorts as pandas dataframes
    'nlst.train_cohort': lambda: NLST_CohortWrapper().train_cohort,
    'nlst.ft_train': lambda: NLST_CohortWrapper().ft_train,
    'nlst.test_scan': lambda: NLST_CohortWrapper().test_scan,
    'nlst.ft_test_scan': lambda: NLST_CohortWrapper().ft_test_scan,
}

def cli_main():
    parser = argparse.ArgumentParser()
    parser.add_argument("config", type=str)
    parser.add_argument("phase", choices=['train', 'val', 'cv', 'test', 'predict', 'bootstrap', 'bootstrap_group'])
    parser.add_argument("cohort", choices=COHORTS.keys())
    # parser.add_argument("--label", type=str, default=None)
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--data_root", type=str, default=None)
    parser.add_argument("--suffix", type=str, default=None) # suffix of filename for test predictions
    parser.add_argument("--predict_dst", type=str, default=None) # predict: save latent representations to this location
    parser.add_argument("--predictions", type=str, default=None) # bootstrap: compute CI from this predictions file
    args = parser.parse_args() 

    L.seed_everything(definitions.RANDOM_SEED)
    torch.set_float32_matmul_precision('high')
    config = Config(args.config)
    cohort_df = COHORTS[args.cohort]()

    if args.phase == 'train':
        trainer = trainer_from_config(config, "train")

        # DataModule (lightning data module)
        datamodule = datamodule_from_config(cohort_df, config)

        # Model (lightniing module)
        total_steps = math.ceil(len(datamodule.dataframe)*config.val_split/config.batch_size)
        Lmodel = LMODELS[config.lmodel](config, total_steps)
        train(datamodule=datamodule, Lmodel=Lmodel, trainer=trainer, checkpoint=args.checkpoint)
    
    elif args.phase == 'test':
        suffix = args.cohort.split(".")[-1] if args.suffix is None else args.suffix
        trainer = trainer_from_config(config, "test", suffix=suffix, predict_dst=args.predict_dst)

        # DataModule (lightning data module)
        datamodule = datamodule_from_config(cohort_df, config)

        # Model (lightniing module)
        Lmodel = LMODELS[config.lmodel](config)
        checkpoint = args.checkpoint if args.checkpoint else checkpoint_from_config(config)
        # assert checkpoint is not None, "Must specify checkpoint"
        test(datamodule=datamodule, Lmodel=Lmodel, trainer=trainer, checkpoint=checkpoint)
    
    elif args.phase == "predict":
        # save latent representations
        assert args.predict_dst is not None, "Must specify --predict_dst"
        trainer = trainer_from_config(config, "predict", suffix=args.suffix or "", predict_dst=args.predict_dst)

        # DataModule (lightning data module)
        datamodule = datamodule_from_config(cohort_df, config)

        # Model (lightniing module)
        Lmodel = LMODELS[config.lmodel](config)
        # checkpoint = args.checkpoint if args.checkpoint else checkpoint_from_config(config)
        # assert checkpoint is not None, "Must specify checkpoint"
        predict(datamodule=datamodule, Lmodel=Lmodel, trainer=trainer, checkpoint=None)
        
    elif args.phase == "bootstrap":
        if args.predictions is None:
            fname = f"test_pred{args.suffix}.csv" if args.suffix else "test_pred.csv"
            pred_df = read_tabular(os.path.join(definitions.DATA_DIR, config.id, fname))
        else:
            pred_df = read_tabular(args.predictions)
        metrics = bootstrap(
            pred_df,
            agg=lambda x: auc_from_df(x, cols=('label', 'prob')),
        )
        print(metrics)
    
    elif args.phase == "bootstrap_group":
        # compute boostrap CI for each group
        dtype_args = {'dtype': {'pid': str, 'scandate': str}}
        if args.predictions is None:
            fname = f"test_pred{args.suffix}.csv" if args.suffix else "test_pred.csv"
            pred_df = read_tabular(os.path.join(definitions.DATA_DIR, config.id, fname), fargs=dtype_args)
        else:
            pred_df = read_tabular(args.predictions, fargs=dtype_args)
        cohort_df['scandate'] = cohort_df['scandate'].apply(lambda x: format_datetime_str(x))
        group_pred = pred_df.merge(cohort_df, on=['pid', 'scandate'])
        metrics = bootstrap(
            group_pred,
            agg=lambda x: auc_from_df(x, cols=('label', 'prob')),
            grps=['institute']
        )
        print(metrics)

if __name__ == "__main__":
    cli_main()
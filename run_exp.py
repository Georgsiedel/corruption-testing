import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "True"

if __name__ == '__main__':
    import numpy as np
    import importlib
    from experiments.eval import eval_metric
    import experiments.utils as utils
    import pandas as pd

    os.environ["CUDA_LAUNCH_BLOCKING"] = "1" #prevents "CUDA error: unspecified launch failure" and is recommended for some illegal memory access errors #increases train time by ~5-15%
    #os.environ["CUDA_VISIBLE_DEVICES"] = "1" #this blocks the spawn of multiple workers

    for experiment in [9,2]:

        configname = (f'experiments.configs.config{experiment}')
        config = importlib.import_module(configname)

        print('Starting experiment #',experiment, 'on', config.dataset, 'dataset')
        runs = 1

        if experiment == 9:
            resume = True
        else:
            resume = False

        for run in range(runs):
            print("Training run #",run)
            if config.combine_train_corruptions:
                print('Combined training')
                cmd0 = "python experiments/train.py --resume={} --run={} --experiment={} --epochs={} --learningrate={} --dataset={} " \
                       "--validontest={} --lrschedule={} --lrparams=\"{}\" --earlystop={} --earlystopPatience={} --optimizer={} " \
                       "--optimizerparams=\"{}\" --modeltype={} --modelparams=\"{}\" --resize={} --aug_strat_check={} " \
                       "--train_aug_strat={} --loss={} --lossparams=\"{}\" --trades_loss={} --trades_lossparams=\"{}\" " \
                       "--robust_loss={} --robust_lossparams=\"{}\" --mixup=\"{}\" " \
                       "--cutmix=\"{}\" --manifold=\"{}\" --combine_train_corruptions={} " \
                       "--concurrent_combinations={} --batchsize={} --number_workers={} " \
                       "--RandomEraseProbability={} --warmupepochs={} --normalize={} --pixel_factor={} --minibatchsize={} " \
                       "--validonc={} --validonadv={} --swa={} --noise_sparsity={} --noise_patch_lower_scale={}"\
                    .format(resume, run, experiment, config.epochs, config.learningrate, config.dataset, config.validontest,
                            config.lrschedule, config.lrparams, config.earlystop, config.earlystopPatience,
                            config.optimizer, config.optimizerparams, config.modeltype, config.modelparams, config.resize,
                            config.aug_strat_check, config.train_aug_strat, config.loss, config.lossparams, config.trades_loss,
                            config.trades_lossparams, config.robust_loss, config.robust_lossparams,config.mixup,
                            config.cutmix, config.manifold, config.combine_train_corruptions, config.concurrent_combinations,
                            config.batchsize, config.number_workers, config.RandomEraseProbability,
                            config.warmupepochs, config.normalize, config.pixel_factor, config.minibatchsize,
                            config.validonc, config.validonadv, config.swa, config.noise_sparsity, config.noise_patch_lower_scale)
                os.system(cmd0)
            else:
                for id, (train_corruption) in enumerate(config.train_corruptions):
                    print("Separate corruption training:", train_corruption)
                    cmd0 = "python experiments/train.py --resume={} --train_corruptions=\"{}\" --run={} --experiment={} " \
                           "--epochs={} --learningrate={} --dataset={} --validontest={} --lrschedule={} --lrparams=\"{}\" " \
                           "--earlystop={} --earlystopPatience={} --optimizer={} --optimizerparams=\"{}\" --modeltype={} " \
                           "--modelparams=\"{}\" --resize={} --aug_strat_check={} --train_aug_strat={} --loss={} " \
                           "--lossparams=\"{}\" --trades_loss={} --trades_lossparams=\"{}\" --robust_loss={} " \
                           "--robust_lossparams=\"{}\" --mixup=\"{}\" --cutmix=\"{}\" --manifold=\"{}\" " \
                           "--combine_train_corruptions={} --concurrent_combinations={} --batchsize={} --number_workers={} " \
                           "--RandomEraseProbability={} --warmupepochs={} --normalize={} --pixel_factor={} " \
                           "--minibatchsize={} --validonc={} --validonadv={} --swa={} --noise_sparsity={} --noise_patch_lower_scale={}"\
                        .format(resume, train_corruption, run, experiment, config.epochs, config.learningrate,
                                config.dataset, config.validontest, config.lrschedule, config.lrparams, config.earlystop,
                                config.earlystopPatience, config.optimizer, config.optimizerparams, config.modeltype,
                                config.modelparams, config.resize, config.aug_strat_check, config.train_aug_strat,
                                config.loss, config.lossparams, config.trades_loss,
                                config.trades_lossparams, config.robust_loss, config.robust_lossparams, config.mixup,
                                config.cutmix, config.manifold, config.combine_train_corruptions,
                                config.concurrent_combinations, config.batchsize, config.number_workers,
                                config.RandomEraseProbability, config.warmupepochs, config.normalize, config.pixel_factor,
                                config.minibatchsize, config.validonc, config.validonadv, config.swa, config.noise_sparsity,
                                config.noise_patch_lower_scale)
                    #if experiment == 9:
                    #    print('skip')
                    #else:
                    os.system(cmd0)

        # Calculate accuracy and robust accuracy, evaluating each trained network on each corruption
        print('Beginning metric evaluation')
        all_test_metrics = np.empty([config.test_count, config.model_count, runs])
        avg_test_metrics = np.empty([config.test_count, config.model_count])
        std_test_metrics = np.empty([config.test_count, config.model_count])
        max_test_metrics = np.empty([config.test_count, config.model_count])

        for run in range(runs):
            print("Evaluation run #",run)
            test_metrics = np.empty([config.test_count, config.model_count])

            if config.combine_train_corruptions:
                print("Evaluating model of combined type")
                filename = f'./experiments/trained_models/{config.dataset}/{config.modeltype}/config{experiment}_' \
                           f'{config.lrschedule}_combined_run_{run}.pth'
                test_metric_col, adv_fig, clever_scores_sorted, adv_distance_sorted1 = eval_metric(filename,
                                            config.test_corruptions, config.combine_test_corruptions, config.test_on_c,
                                            config.modeltype, config.modelparams, config.resize, config.dataset, 1000,
                                            config.number_workers, config.normalize, config.calculate_adv_distance, config.adv_distance_params,
                                            config.calculate_autoattack_robustness, config.autoattack_params, config.pixel_factor)
                test_metrics[:, 0] = np.array(test_metric_col)
                print(test_metric_col)
                if adv_fig:
                    adv_fig.savefig(f'results/{config.dataset}/{config.modeltype}/'
                                    f'config{experiment}_{config.lrschedule}_combined_adversarial_distance_run_{run}.svg')
                adv_distance_frame = pd.DataFrame({"Adversarial_Distance_sorted": adv_distance_sorted1,
                                                   "Clever_Score_sorted_by_Adversarial_Distance": clever_scores_sorted})
                adv_distance_frame.to_csv(f'./results/{config.dataset}/{config.modeltype}/config{experiment}_'
                                          f'{config.lrschedule}_combined_adversarial_distance_run_{run}.csv',
                    index=False, header=True, sep=';', float_format='%1.4f', decimal=',')
            else:
                for idx, (train_corruption) in enumerate(config.train_corruptions):
                    print("Evaluating model trained on corruption of type:", train_corruption)
                    filename = f'./experiments/trained_models/{config.dataset}/{config.modeltype}/config{experiment}_' \
                               f'{config.lrschedule}_separate_{train_corruption["noise_type"]}_eps_{train_corruption["epsilon"]}_{train_corruption["sphere"]}_run_{run}.pth'
                    test_metric_col, adv_fig, clever_scores_sorted, adv_distance_sorted1 = eval_metric(filename,
                                                config.test_corruptions, config.combine_test_corruptions, config.test_on_c,
                                                config.modeltype, config.modelparams, config.resize, config.dataset, 1000,
                                                config.number_workers, config.normalize, config.calculate_adv_distance, config.adv_distance_params,
                                                config.calculate_autoattack_robustness, config.autoattack_params, config.pixel_factor)
                    test_metrics[:, idx] = np.array(test_metric_col)
                    print(test_metric_col)
                    if adv_fig:
                        adv_fig.savefig(f'results/{config.dataset}/{config.modeltype}/config{experiment}_{config.lrschedule}_'
                                        f'separate_adversarial_distance_{train_corruption["noise_type"]}_eps_{train_corruption["epsilon"]}_{train_corruption["sphere"]}_run_{run}.svg')
                    adv_distance_frame = pd.DataFrame({"Adversarial_Distance_sorted": adv_distance_sorted1,
                                                       "Clever_Score_sorted_by_Adversarial_Distance": clever_scores_sorted})
                    adv_distance_frame.to_csv(f'./results/{config.dataset}/{config.modeltype}/config{experiment}_{config.lrschedule}_'
                                        f'separate_adversarial_distance_{train_corruption["noise_type"]}_eps_{train_corruption["epsilon"]}_{train_corruption["sphere"]}_run_{run}.csv',
                        index=False, header=True, sep=';', float_format='%1.4f', decimal=',')

            all_test_metrics[:config.test_count, :config.model_count, run] = test_metrics

        for idm in range(config.model_count):
            for ide in range(config.test_count):
                avg_test_metrics[ide, idm] = all_test_metrics[ide, idm, :].mean()
                std_test_metrics[ide, idm] = all_test_metrics[ide, idm, :].std()
                max_test_metrics[ide, idm] = all_test_metrics[ide, idm, :].max()

        utils.create_report(avg_test_metrics, max_test_metrics, std_test_metrics, config.train_corruptions, config.test_corruptions,
                      config.combine_train_corruptions, config.combine_test_corruptions, config.dataset, config.modeltype,
                      config.lrschedule, experiment, config.test_on_c, config.calculate_adv_distance,
                      config.calculate_autoattack_robustness, runs)
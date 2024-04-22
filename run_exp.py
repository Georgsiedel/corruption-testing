import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "True"

if __name__ == '__main__':
    import importlib

    os.environ["CUDA_LAUNCH_BLOCKING"] = "1" #prevents "CUDA error: unspecified launch failure" and is recommended for some illegal memory access errors #increases train time by ~5-15%
    #os.environ["CUDA_VISIBLE_DEVICES"] = "1" #this blocks the spawn of multiple workers

    for experiment in [9,2]:

        configname = (f'experiments.configs.config{experiment}')
        config = importlib.import_module(configname)

        print('Starting experiment #',experiment, 'on', config.dataset, 'dataset')
        runs = 1

        if experiment == 0:
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
                       "--robust_loss={} --robust_lossparams=\"{}\" --mixup=\"{}\" --cutmix=\"{}\" --manifold=\"{}\" " \
                       "--combine_train_corruptions={} --concurrent_combinations={} --batchsize={} --number_workers={} " \
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
                if experiment == (2):
                    print('skip')
                else:
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
                    if experiment == (9):
                        print('skip')
                    else:
                        os.system(cmd0)


        # Calculate accuracy and robust accuracy, evaluating each trained network on each corruption
        print('Beginning metric evaluation')

        cmdeval = "python experiments/eval.py --resume={} --experiment={} --runs={} --batchsize={} --dataset={} " \
                "--modeltype={} --modelparams=\"{}\" --resize={} --combine_test_corruptions={} --number_workers={} " \
                "--normalize={} --pixel_factor={} --test_on_c={} --calculate_adv_distance={} --adv_distance_params=\"{}\" " \
                "--calculate_autoattack_robustness={} --autoattack_params=\"{}\" --combine_train_corruptions={} " \
                .format(resume, experiment, runs, config.batchsize, config.dataset, config.modeltype, config.modelparams,
                        config.resize, config.combine_test_corruptions, config.number_workers, config.normalize,
                        config.pixel_factor, config.test_on_c, config.calculate_adv_distance, config.adv_distance_params,
                        config.calculate_autoattack_robustness, config.autoattack_params, config.combine_train_corruptions)
        os.system(cmdeval)

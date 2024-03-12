import numpy as np

train_corruptions = np.array([
#{'noise_type': 'standard', 'epsilon': 0.0, 'sphere': False, 'distribution': 'beta2-5'},
{'noise_type': 'uniform-linf', 'epsilon': 0.15, 'sphere': False, 'distribution': 'uniform'},
{'noise_type': 'gaussian', 'epsilon': 0.15, 'sphere': False, 'distribution': 'uniform'},
{'noise_type': 'uniform-l0.5', 'epsilon': 400000.0, 'sphere': False, 'distribution': 'uniform'},
{'noise_type': 'uniform-l1', 'epsilon': 200.0, 'sphere': False, 'distribution': 'uniform'},
{'noise_type': 'uniform-l2', 'epsilon': 5.0, 'sphere': False, 'distribution': 'uniform'},
{'noise_type': 'uniform-l5', 'epsilon': 0.6, 'sphere': False, 'distribution': 'uniform'},
{'noise_type': 'uniform-l50', 'epsilon': 0.18, 'sphere': False, 'distribution': 'uniform'},
{'noise_type': 'uniform-l0-impulse', 'epsilon': 0.2, 'sphere': False, 'distribution': 'uniform'},
{'noise_type': 'uniform-l0-impulse', 'epsilon': 0.15, 'sphere': True, 'distribution': 'uniform'}
])
noise_sparsity = 0.5
noise_patch_lower_scale = 0.5
combine_train_corruptions = True #augment the train dataset with all corruptions
concurrent_combinations = 1 #only has an effect if combine_train_corruption is True

batchsize = 256
minibatchsize = 8
dataset = 'CIFAR100' #ImageNet #CIFAR100 #CIFAR10 #TinyImageNet
normalize = True
validontest = True
validonc = True
lrschedule = 'CosineAnnealingWarmRestarts'
learningrate = 0.2
epochs = 450
lrparams = {'T_0': 30, 'T_mult': 2}
warmupepochs = 0
earlystop = False
earlystopPatience = 15
optimizer = 'SGD'
optimizerparams = {'momentum': 0.9, 'weight_decay': 1e-4}
number_workers = 1
modeltype = 'WideResNet_28_4'
modelparams = {'dropout_rate': 0.2, 'activation_function': 'silu'}
resize = False
aug_strat_check = True
train_aug_strat = 'TrivialAugmentWide' #TrivialAugmentWide, RandAugment, AutoAugment, AugMix
loss_function = 'jsd' #'ce', 'jsd'
lossparams = {'num_splits': 3, 'alpha': 12, 'smoothing': 0.1}
mixup = {'alpha': 0.2, 'p': 1.0} #default alpha 0.2 #If both mixup and cutmix are >0, mixup or cutmix are selected by 0.5 chance
cutmix = {'alpha': 1.0, 'p': 1.0} # default alpha 1.0 #If both mixup and cutmix are >0, mixup or cutmix are selected by 0.5 chance
manifold = {'apply': True, 'noise_factor': 2}
RandomEraseProbability = 0.0
swa = False


#define train and test corruptions:
#define noise type (first column): 'gaussian', 'uniform-l0-impulse', 'uniform-l0-salt-pepper', 'uniform-linf'. also: all positive numbers p>0 for uniform Lp possible: 'uniform-l1', 'uniform-l2', ...
#define intensity (second column): max.-distance of random perturbations for model training and evaluation (gaussian: std-dev; l0: proportion of pixels corrupted; lp: epsilon)
#define whether density_distribution=max (third column) is True (sample only maximum intensity values) or False (uniformly distributed up to maximum intensity)
test_corruptions = np.array([
{'noise_type': 'uniform-l0-impulse', 'epsilon': 0.01, 'sphere': True, 'distribution': 'max'},
{'noise_type': 'uniform-l0-impulse', 'epsilon': 0.02, 'sphere': True, 'distribution': 'max'},
{'noise_type': 'uniform-l0-impulse', 'epsilon': 0.05, 'sphere': True, 'distribution': 'max'},
{'noise_type': 'uniform-l0-impulse', 'epsilon': 0.1, 'sphere': True, 'distribution': 'max'},
{'noise_type': 'uniform-l0-impulse', 'epsilon': 0.2, 'sphere': True, 'distribution': 'max'},
{'noise_type': 'uniform-l0.5', 'epsilon': 50000.0, 'sphere': False, 'distribution': 'max'},
{'noise_type': 'uniform-l0.5', 'epsilon': 100000.0, 'sphere': False, 'distribution': 'max'},
{'noise_type': 'uniform-l0.5', 'epsilon': 200000.0, 'sphere': False, 'distribution': 'max'},
{'noise_type': 'uniform-l0.5', 'epsilon': 300000.0, 'sphere': False, 'distribution': 'max'},
{'noise_type': 'uniform-l0.5', 'epsilon': 500000.0, 'sphere': False, 'distribution': 'max'},
{'noise_type': 'uniform-l1', 'epsilon': 25.0, 'sphere': False, 'distribution': 'max'},
{'noise_type': 'uniform-l1', 'epsilon': 50.0, 'sphere': False, 'distribution': 'max'},
{'noise_type': 'uniform-l1', 'epsilon': 100.0, 'sphere': False, 'distribution': 'max'},
{'noise_type': 'uniform-l1', 'epsilon': 200.0, 'sphere': False, 'distribution': 'max'},
{'noise_type': 'uniform-l1', 'epsilon': 300.0, 'sphere': False, 'distribution': 'max'},
{'noise_type': 'uniform-l2', 'epsilon': 0.5, 'sphere': False, 'distribution': 'max'},
{'noise_type': 'uniform-l2', 'epsilon': 1.0, 'sphere': False, 'distribution': 'max'},
{'noise_type': 'uniform-l2', 'epsilon': 2.0, 'sphere': False, 'distribution': 'max'},
{'noise_type': 'uniform-l2', 'epsilon': 3.0, 'sphere': False, 'distribution': 'max'},
{'noise_type': 'uniform-l2', 'epsilon': 5.0, 'sphere': False, 'distribution': 'max'},
{'noise_type': 'uniform-l5', 'epsilon': 0.05, 'sphere': False, 'distribution': 'max'},
{'noise_type': 'uniform-l5', 'epsilon': 0.1, 'sphere': False, 'distribution': 'max'},
{'noise_type': 'uniform-l5', 'epsilon': 0.2, 'sphere': False, 'distribution': 'max'},
{'noise_type': 'uniform-l5', 'epsilon': 0.4, 'sphere': False, 'distribution': 'max'},
{'noise_type': 'uniform-l5', 'epsilon': 0.8, 'sphere': False, 'distribution': 'max'},
{'noise_type': 'uniform-l10', 'epsilon': 0.02, 'sphere': False, 'distribution': 'max'},
{'noise_type': 'uniform-l10', 'epsilon': 0.05, 'sphere': False, 'distribution': 'max'},
{'noise_type': 'uniform-l10', 'epsilon': 0.1, 'sphere': False, 'distribution': 'max'},
{'noise_type': 'uniform-l10', 'epsilon': 0.2, 'sphere': False, 'distribution': 'max'},
{'noise_type': 'uniform-l10', 'epsilon': 0.4, 'sphere': False, 'distribution': 'max'},
{'noise_type': 'uniform-l50', 'epsilon': 0.02, 'sphere': False, 'distribution': 'max'},
{'noise_type': 'uniform-l50', 'epsilon': 0.05, 'sphere': False, 'distribution': 'max'},
{'noise_type': 'uniform-l50', 'epsilon': 0.1, 'sphere': False, 'distribution': 'max'},
{'noise_type': 'uniform-l50', 'epsilon': 0.15, 'sphere': False, 'distribution': 'max'},
{'noise_type': 'uniform-l50', 'epsilon': 0.2, 'sphere': False, 'distribution': 'max'},
{'noise_type': 'uniform-l200', 'epsilon': 0.02, 'sphere': False, 'distribution': 'max'},
{'noise_type': 'uniform-l200', 'epsilon': 0.04, 'sphere': False, 'distribution': 'max'},
{'noise_type': 'uniform-l200', 'epsilon': 0.08, 'sphere': False, 'distribution': 'max'},
{'noise_type': 'uniform-l200', 'epsilon': 0.12, 'sphere': False, 'distribution': 'max'},
{'noise_type': 'uniform-l200', 'epsilon': 0.2, 'sphere': False, 'distribution': 'max'},
{'noise_type': 'uniform-linf', 'epsilon': 0.01, 'sphere': False, 'distribution': 'max'},
{'noise_type': 'uniform-linf', 'epsilon': 0.02, 'sphere': False, 'distribution': 'max'},
{'noise_type': 'uniform-linf', 'epsilon': 0.05, 'sphere': False, 'distribution': 'max'},
{'noise_type': 'uniform-linf', 'epsilon': 0.1, 'sphere': False, 'distribution': 'max'},
{'noise_type': 'uniform-linf', 'epsilon': 0.2, 'sphere': False, 'distribution': 'max'},
{'noise_type': 'gaussian', 'epsilon': 0.01, 'sphere': False, 'distribution': 'max'},
{'noise_type': 'gaussian', 'epsilon': 0.02, 'sphere': False, 'distribution': 'max'},
{'noise_type': 'gaussian', 'epsilon': 0.05, 'sphere': False, 'distribution': 'max'},
{'noise_type': 'gaussian', 'epsilon': 0.1, 'sphere': False, 'distribution': 'max'},
{'noise_type': 'gaussian', 'epsilon': 0.2, 'sphere': False, 'distribution': 'max'}
])

test_on_c = True
combine_test_corruptions = False #augment the test dataset with all corruptions
calculate_adv_distance = False
adv_distance_params = {'setsize': 1000, 'nb_iters': 100, 'eps_iter': 0.0005, 'norm': np.inf, "epsilon": 0.1,
                       "clever": True, "clever_batches": 500, "clever_samples": 1024}
calculate_autoattack_robustness = False
autoattack_params = {'setsize': 1000, 'epsilon': 8/255, 'norm': 'Linf'}


if combine_train_corruptions:
    model_count = 1
else:
    model_count = train_corruptions.shape[0]
if dataset == 'CIFAR10':
    num_classes = 10
    pixel_factor = 1
elif dataset == 'CIFAR100':
    num_classes = 100
    pixel_factor = 1
elif dataset == 'ImageNet':
    num_classes = 1000
elif dataset == 'TinyImageNet':
    num_classes = 200
    pixel_factor = 2


test_count = 2
if test_on_c:
    test_count += 23
if combine_test_corruptions:
    test_count += 1
else:
    test_count += test_corruptions.shape[0]
if calculate_adv_distance:
    test_count += 4
if calculate_autoattack_robustness:
    test_count += 2
import numpy as np

train_corruptions = np.array([
#{'noise_type': 'standard', 'epsilon': 0.0, 'sphere': False, 'distribution': 'beta2-5'},
{'noise_type': 'uniform-linf', 'epsilon': 0.3, 'sphere': False, 'distribution': 'uniform'},
{'noise_type': 'gaussian', 'epsilon': 0.2, 'sphere': False, 'distribution': 'uniform'},
{'noise_type': 'uniform-l0.5', 'epsilon': 12000000.0, 'sphere': False, 'distribution': 'uniform'},
{'noise_type': 'uniform-l1', 'epsilon': 1500.0, 'sphere': False, 'distribution': 'uniform'},
{'noise_type': 'uniform-l2', 'epsilon': 20.0, 'sphere': False, 'distribution': 'uniform'},
{'noise_type': 'uniform-l5', 'epsilon': 1.5, 'sphere': False, 'distribution': 'uniform'},
{'noise_type': 'uniform-l50', 'epsilon': 0.35, 'sphere': False, 'distribution': 'uniform'},
{'noise_type': 'uniform-l0-impulse', 'epsilon': 0.35, 'sphere': False, 'distribution': 'uniform'},
{'noise_type': 'uniform-l0-impulse', 'epsilon': 0.3, 'sphere': True, 'distribution': 'uniform'}
])
noise_sparsity = 0.0
combine_train_corruptions = True #augment the train dataset with all corruptions
concurrent_combinations = 1 #only has an effect if combine_train_corruption is True

batchsize = 384
minibatchsize = 8
dataset = 'TinyImageNet' #ImageNet #CIFAR100 #CIFAR10 #TinyImageNet
normalize = True
validontest = True
validonc = True
lrschedule = 'CosineAnnealingWarmRestarts'
learningrate = 0.1
epochs = 150
lrparams = {'T_0': 10, 'T_mult': 2}
warmupepochs = 0
earlystop = False
earlystopPatience = 15
optimizer = 'SGD'
optimizerparams = {'momentum': 0.9, 'weight_decay': 5e-4}
number_workers = 1
modeltype = 'WideResNet_28_4'
modelparams = {'dropout_rate': 0.3}
resize = False
aug_strat_check = True
train_aug_strat = 'TrivialAugmentWide' #TrivialAugmentWide, RandAugment, AutoAugment, AugMix
loss_function = 'ce' #'ce', 'jsd'
lossparams = {'num_splits': 3, 'alpha': 12, 'smoothing': 0.1}
mixup = {'alpha': 0.2, 'p': 0.0} #default alpha 0.2 #If both mixup and cutmix are >0, mixup or cutmix are selected by 0.5 chance
cutmix = {'alpha': 1.0, 'p': 0.0} # default alpha 1.0 #If both mixup and cutmix are >0, mixup or cutmix are selected by 0.5 chance
manifold = {'apply': False, 'noise_factor': 2}
RandomEraseProbability = 0.0
swa = False

#define train and test corruptions:
#define noise type (first column): 'gaussian', 'uniform-l0-impulse', 'uniform-l0-salt-pepper', 'uniform-linf'. also: all positive numbers p>0 for uniform Lp possible: 'uniform-l1', 'uniform-l2', ...
#define intensity (second column): max.-distance of random perturbations for model training and evaluation (gaussian: std-dev; l0: proportion of pixels corrupted; lp: epsilon)
#define whether density_distribution=max (third column) is True (sample only maximum intensity values) or False (uniformly distributed up to maximum intensity)
test_corruptions = np.array([
{'noise_type': 'uniform-l0-impulse', 'epsilon': 0.1, 'sphere': True, 'distribution': 'max'},
{'noise_type': 'uniform-linf', 'epsilon': 0.15, 'sphere': False, 'distribution': 'max'},
{'noise_type': 'gaussian', 'epsilon': 0.1, 'sphere': False, 'distribution': 'max'}
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